"""Bounded pick/place state machine. No ROS or simulator truth dependencies.

The backend supplies calibrated RGB-D observations and measured motion feedback.
Missing/ambiguous/stale evidence is a failure, never evidence of a successful grasp.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
import time


class State(str, Enum):
    HOME = 'HOME'
    ACQUIRE = 'ACQUIRE'
    APPROACH = 'APPROACH'
    REFINE = 'REFINE'
    DESCEND = 'DESCEND'
    CLOSE = 'CLOSE'
    LIFT = 'LIFT'
    VERIFY_LIFT = 'VERIFY_LIFT'
    TRANSFER = 'TRANSFER'
    PLACE = 'PLACE'
    RELEASE = 'RELEASE'
    RETREAT = 'RETREAT'
    VERIFY_PLACE = 'VERIFY_PLACE'
    RECOVER = 'RECOVER'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'


@dataclass
class Observation:
    position: list
    appearance: list
    label: str
    confidence: float
    stamp_ns: int
    received_at: float
    quat: list = field(default_factory=lambda: [1., 0., 0., 0.])
    extent_xy: list = None
    source: str = 'unspecified'


@dataclass
class PolicyConfig:
    optimized_paths: bool = True
    target_label: str = None
    acquire_camera: str = 'wrist'
    max_attempts: int = 2
    observation_retries: int = 3
    verification_frames: int = 2
    association_distance: float = 0.075
    appearance_distance: float = 0.35
    ambiguity_margin: float = 0.01
    max_observation_age: float = 8.0
    approach_clearance: float = 0.18
    viewpoint_offset_xy: list = field(default_factory=lambda: [-0.08,0.0])
    observation_mode: str = 'camera_centered'
    camera_clearance: float = 0.38
    min_refinement_area_ratio: float = 0.55
    tcp_offset: float = 0.20
    lift_distance: float = 0.12
    min_lift: float = 0.06
    place_tolerance_xy: float = 0.06
    place_tolerance_z: float = 0.035
    stability_tolerance: float = 0.015
    safe_z: float = 0.50
    final_approach_clearance: float = .08
    retreat_clearance: float = .12
    home: list = field(default_factory=lambda: [0.30, 0.25, 0.50])
    place_surface: list = field(default_factory=lambda: [0.35, -0.35, 0.05])
    workspace_min: list = field(default_factory=lambda: [-0.7, -0.7, -0.02])
    workspace_max: list = field(default_factory=lambda: [0.7, 0.7, 0.75])

    def __post_init__(self):
        if not 1 <= self.max_attempts <= 5 or not 1 <= self.observation_retries <= 10:
            raise ValueError('Retry limits must be bounded')
        if self.verification_frames < 2:
            raise ValueError('Verification needs at least two distinct frames')
        if self.observation_mode not in ['camera_centered','legacy_offset']:
            raise ValueError('Unknown observation mode')
        if not math.isfinite(self.camera_clearance) or self.camera_clearance <= 0.:
            raise ValueError('Camera clearance must be positive')


class PolicyFailure(RuntimeError):
    pass


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


class TargetLock:
    """Keep the original appearance; updates never gradually change identity."""
    def __init__(self, observation, config):
        self.original = observation
        self.config = config
        self.last_stamp = observation.stamp_ns

    def match(self, observations, expected):
        plausible = []
        for obs in observations:
            if obs.stamp_ns <= self.last_stamp:
                continue
            if obs.label != self.original.label:
                continue
            if time.monotonic() - obs.received_at > self.config.max_observation_age:
                continue
            if not all(math.isfinite(v) for v in obs.position + obs.appearance):
                continue
            if distance(obs.appearance, self.original.appearance) > self.config.appearance_distance:
                continue
            d = distance(obs.position, expected)
            if d <= self.config.association_distance:
                plausible.append((d, obs))
        plausible.sort(key=lambda pair: pair[0])
        if not plausible:
            raise PolicyFailure('target_missing_or_stale')
        if len(plausible) > 1 and plausible[1][0] - plausible[0][0] < self.config.ambiguity_margin:
            raise PolicyFailure('target_ambiguous')
        chosen = plausible[0][1]
        self.last_stamp = chosen.stamp_ns
        return chosen


class FeedbackPolicy:
    def __init__(self, backend, config=None, event_sink=None):
        self.backend = backend
        self.cfg = config or PolicyConfig()
        self.event_sink = event_sink
        self.events = []
        self.state = None
        self.attempt = 0
        self.lock = None
        self.may_be_holding = False
        self.lift_verified = False

    def emit(self, state, **details):
        self.state = state
        event = dict(state=state.value, attempt=self.attempt, wall_time=time.time(),
                     monotonic=time.monotonic(), **details)
        self.events.append(event)
        if self.event_sink:
            self.event_sink(event)

    def move(self, state, position, quat=None, linear=False):
        self.emit(state, goal=position)
        if hasattr(self.backend,'set_motion_phase'):
            self.backend.set_motion_phase(state.value)
        if not self.backend.move(position, quat or [1., 0., 0., 0.], linear=linear):
            raise PolicyFailure('motion_failed:' + state.value)

    def acquire(self):
        self.emit(State.ACQUIRE)
        for _ in range(self.cfg.observation_retries):
            observations = self.backend.observe(self.cfg.acquire_camera)
            valid = [o for o in observations if
                     (self.cfg.target_label is None or o.label == self.cfg.target_label) and
                     len(o.position) == 3 and all(math.isfinite(v) for v in o.position + o.appearance) and
                     time.monotonic() - o.received_at <= self.cfg.max_observation_age and
                     all(lo <= v <= hi for lo, v, hi in zip(self.cfg.workspace_min, o.position, self.cfg.workspace_max))]
            if valid:
                target = max(valid, key=lambda o: o.confidence)
                self.lock = TargetLock(target, self.cfg)
                return target
        raise PolicyFailure('no_valid_target')

    def observe_locked(self, camera, expected):
        if hasattr(self.backend,'set_tracking_target'):
            self.backend.set_tracking_target(expected,self.lock.original)
        last_error = 'target_missing'
        for _ in range(self.cfg.observation_retries):
            try:
                return self.lock.match(self.backend.observe(camera), expected)
            except PolicyFailure as exc:
                last_error = str(exc)
        raise PolicyFailure(last_error)

    def verify(self, state, expected, baseline_z=None):
        self.emit(state, expected=expected)
        previous = None
        for _ in range(self.cfg.verification_frames):
            obs = self.observe_locked('observer', expected)
            if state == State.VERIFY_LIFT:
                if obs.position[2] - baseline_z < self.cfg.min_lift:
                    raise PolicyFailure('object_not_lifted')
            else:
                if distance(obs.position[:2], expected[:2]) > self.cfg.place_tolerance_xy:
                    raise PolicyFailure('placement_xy_error')
                if abs(obs.position[2] - expected[2]) > self.cfg.place_tolerance_z:
                    raise PolicyFailure('placement_height_error')
            if previous is not None and distance(previous.position, obs.position) > self.cfg.stability_tolerance:
                raise PolicyFailure('object_not_stable')
            previous = obs
            self.emit(state, evidence=asdict(obs))
        return previous

    def run(self):
        started = time.monotonic()
        reason = ''
        try:
            self.move(State.HOME, self.cfg.home)
            if not self.backend.grip(False):
                raise PolicyFailure('gripper_open_failed')
            target = self.acquire()
            self.emit(State.ACQUIRE, locked_target=asdict(target))
            for attempt in range(1, self.cfg.max_attempts + 1):
                self.attempt = attempt
                try:
                    p, q = target.position, target.quat
                    if self.cfg.observation_mode == 'camera_centered':
                        above,view_quat=self.backend.observation_pose(p,self.cfg.camera_clearance)
                    else:
                        above=[p[0]+self.cfg.viewpoint_offset_xy[0],p[1]+self.cfg.viewpoint_offset_xy[1],
                               p[2]+self.cfg.tcp_offset+self.cfg.approach_clearance]
                        view_quat=q
                    self.move(State.APPROACH, above, view_quat)
                    self.emit(State.REFINE)
                    refined = self.observe_locked('wrist', p)
                    if target.extent_xy and refined.extent_xy:
                        area0 = math.prod(target.extent_xy)
                        area1 = math.prod(refined.extent_xy)
                        if area0 > 1e-6 and area1/area0 < self.cfg.min_refinement_area_ratio:
                            raise PolicyFailure('refinement_incomplete_mask')
                    target = refined
                    p, q = target.position, target.quat
                    aligned_z = (p[2]+self.cfg.tcp_offset+self.cfg.final_approach_clearance
                                 if self.cfg.optimized_paths else
                                 max(above[2],p[2]+self.cfg.tcp_offset+self.cfg.approach_clearance))
                    aligned = [p[0], p[1], aligned_z]
                    self.move(State.APPROACH, aligned, q)
                    pick = [p[0], p[1], p[2]+self.cfg.tcp_offset]
                    self.move(State.DESCEND, pick, q, linear=True)
                    self.emit(State.CLOSE)
                    self.may_be_holding = True
                    if not self.backend.grip(True):
                        raise PolicyFailure('gripper_close_failed')
                    destination = self.cfg.place_surface
                    travel_z = max(pick[2]+self.cfg.lift_distance,
                        destination[2]+self.cfg.tcp_offset+self.cfg.approach_clearance,self.cfg.safe_z)
                    lifted = [pick[0], pick[1], travel_z if self.cfg.optimized_paths else pick[2]+self.cfg.lift_distance]
                    self.move(State.LIFT, lifted, q, linear=True)
                    expected = [p[0], p[1], p[2]+lifted[2]-pick[2]]
                    self.verify(State.VERIFY_LIFT, expected, p[2])
                    self.lift_verified = True
                    destination = self.cfg.place_surface
                    travel_z = max(lifted[2], destination[2]+self.cfg.tcp_offset+self.cfg.approach_clearance,
                                   self.cfg.safe_z)
                    if not self.cfg.optimized_paths or abs(travel_z-lifted[2]) > .003:
                        self.move(State.TRANSFER, [lifted[0], lifted[1], travel_z], q, linear=True)
                    self.move(State.TRANSFER, [destination[0], destination[1], travel_z], q)
                    self.move(State.PLACE, [destination[0], destination[1], destination[2]+self.cfg.tcp_offset], q, linear=True)
                    self.emit(State.RELEASE)
                    if not self.backend.grip(False):
                        raise PolicyFailure('gripper_release_failed')
                    self.may_be_holding = False
                    retreat_z = (min(travel_z,destination[2]+self.cfg.tcp_offset+self.cfg.retreat_clearance)
                                 if self.cfg.optimized_paths else travel_z)
                    self.move(State.RETREAT, [destination[0], destination[1], retreat_z], q, linear=True)
                    self.verify(State.VERIFY_PLACE, destination)
                    self.emit(State.SUCCESS)
                    return self.result(True, '', started)
                except PolicyFailure as exc:
                    reason = str(exc)
                    failed_state = self.state
                    self.emit(State.RECOVER, reason=reason)
                    # Transport/placement failure may leave an object suspended: stop and hold.
                    if self.lift_verified or reason.startswith('motion_failed'):
                        break
                    if self.may_be_holding:
                        # Retry an empty grasp ONLY with fresh evidence the locked object
                        # is still on its original support surface. Occlusion stays unknown.
                        if failed_state != State.VERIFY_LIFT:
                            break
                        try:
                            table_obs = self.observe_locked('observer', p)
                        except PolicyFailure:
                            break
                        if distance(table_obs.position, p) > self.cfg.stability_tolerance:
                            break
                        if not self.backend.grip(False):
                            break
                        self.may_be_holding = False
                    if attempt == self.cfg.max_attempts:
                        break
                    if not self.backend.grip(False):
                        raise PolicyFailure('recovery_open_failed')
                    self.move(State.HOME, self.cfg.home)
                    target = self.observe_locked('wrist', self.lock.original.position)
        except PolicyFailure as exc:
            reason = str(exc)
        except Exception as exc:
            reason = 'backend_exception:' + type(exc).__name__ + ':' + str(exc)
        try:
            self.backend.hold()
        except Exception as exc:
            reason += ';hold_failed:' + type(exc).__name__
        self.emit(State.FAILED, reason=reason)
        return self.result(False, reason, started)

    def result(self, success, reason, started):
        return dict(success=success, reason=reason, attempts=self.attempt,
                    lift_verified=self.lift_verified, elapsed_s=time.monotonic()-started,
                    events=self.events)
