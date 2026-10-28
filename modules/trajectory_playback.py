"""Validated trajectory playback sampled by the physics loop, independent of render FPS."""
import bisect
import math
import numpy as np


class PhysicsTrajectory:
    def __init__(self, names, times, positions, start, limits, velocities=None):
        self.names = list(names)
        self.times = np.asarray(times, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.start = float(start)
        self.velocities = None if velocities is None else np.asarray(velocities,dtype=float)
        if (not self.names or len(set(self.names)) != len(self.names) or
                self.positions.shape != (len(self.times), len(self.names)) or not len(self.times) or
                not np.isfinite(self.positions).all() or not np.isfinite(self.times).all() or
                self.times[0] < 0 or (len(self.times) > 1 and np.any(np.diff(self.times) <= 0))):
            raise ValueError('Malformed trajectory')
        if self.velocities is not None and (self.velocities.shape != self.positions.shape or
                                           not np.isfinite(self.velocities).all()):
            raise ValueError('Malformed velocities')
        for i, name in enumerate(self.names):
            if name not in limits:
                raise ValueError('Unknown joint')
            lo, hi = limits[name]
            if np.any(self.positions[:, i] < lo-.001) or np.any(self.positions[:, i] > hi+.001):
                raise ValueError('Joint limit violation')
        # Validate Hermite extrema, including possible overshoot between waypoints.
        if self.velocities is not None:
            for i in range(len(self.times)-1):
                h=self.times[i+1]-self.times[i]
                p0,p1=self.positions[i:i+2]
                v0,v1=self.velocities[i:i+2]
                c= h*v0
                b= -3*p0+3*p1-2*h*v0-h*v1
                a= 2*p0-2*p1+h*v0+h*v1
                for j,name in enumerate(self.names):
                    roots=np.roots([3*a[j],2*b[j],c[j]])
                    for root in roots:
                        if abs(root.imag)<1e-10 and 0<float(root.real)<1:
                            u=float(root.real)
                            value=p0[j]+c[j]*u+b[j]*u*u+a[j]*u*u*u
                            lo,hi=limits[name]
                            if value<lo-.001 or value>hi+.001:
                                raise ValueError('Hermite joint limit overshoot')

    def sample(self, now):
        t = max(0., float(now)-self.start)
        if t >= self.times[-1]:
            return self.positions[-1].copy(), True
        i = max(0, bisect.bisect_right(self.times, t)-1)
        if t <= self.times[0]:
            return self.positions[0].copy(), False
        a = (t-self.times[i])/(self.times[i+1]-self.times[i])
        if self.velocities is not None:
            h=self.times[i+1]-self.times[i]
            return ((2*a**3-3*a*a+1)*self.positions[i]+(a**3-2*a*a+a)*h*self.velocities[i]+
                    (-2*a**3+3*a*a)*self.positions[i+1]+(a**3-a*a)*h*self.velocities[i+1]),False
        return (1-a)*self.positions[i]+a*self.positions[i+1], False


def retime_cartesian(path, velocity=.45, acceleration=.8, sample_dt=1/120):
    """Smooth scalar progress along the collision-checked joint polyline.

    No spline overshoot: positions remain on the original path. Numerical bounds
    are checked at physics resolution, with a 10 percent timing margin.
    """
    q = np.asarray(path, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Invalid joint path')
    if velocity <= 0 or acceleration <= 0:
        raise ValueError('Limits must be positive')
    length = np.r_[0., np.cumsum(np.max(np.abs(np.diff(q, axis=0)), axis=1))]
    keep = np.r_[True, np.diff(length) > 1e-10]
    q, length = q[keep], length[keep]
    if len(q) == 1:
        return [0., .1], np.repeat(q, 2, axis=0).tolist()
    duration = max(.3, 1.875*length[-1]/velocity)
    for _ in range(8):
        times = np.linspace(0., duration, max(5, math.ceil(duration/sample_dt)+1))
        u = times/duration
        progress = (10*u**3-15*u**4+6*u**5)*length[-1]
        positions = np.column_stack([np.interp(progress, length, q[:, j]) for j in range(q.shape[1])])
        v = np.diff(positions, axis=0)/(times[1]-times[0])
        a = np.diff(v, axis=0)/(times[1]-times[0])
        factor = max(float(np.max(np.abs(v)))/velocity,
                     math.sqrt(float(np.max(np.abs(a)))/acceleration))
        if factor <= 1.:
            return times.tolist(), positions.tolist()
        duration *= factor*1.1
    raise ValueError('Unable to retime path within limits')
