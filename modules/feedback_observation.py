"""RGB-D geometry, bounded timestamp matching, appearance identity descriptors."""
from collections import OrderedDict
import math
import time
import numpy as np
import cv2


class FrameSynchronizer:
    def __init__(self, capacity=8, max_age=8.):
        self.capacity, self.max_age = capacity, max_age
        self.frames = OrderedDict()

    def add(self, kind, stamp_ns, value, received_at=None):
        entry = self.frames.setdefault(stamp_ns, {})
        entry[kind] = value
        entry.setdefault('received_at', time.monotonic() if received_at is None else received_at)
        while len(self.frames) > self.capacity:
            self.frames.popitem(last=False)

    def latest(self, after_stamp=0, required=('rgb', 'depth', 'calibration')):
        now = time.monotonic()
        for stamp, entry in sorted(self.frames.items(), reverse=True):
            if stamp > after_stamp and all(key in entry for key in required) and now-entry['received_at'] <= self.max_age:
                return stamp, entry
        return None


def mask_geometry(rgb, depth, mask, calibration, return_details=False):
    """Median 3D surface position from valid mask interior; reject missing depth."""
    if rgb.shape[:2] != depth.shape or mask.shape != depth.shape:
        raise ValueError('rgb_depth_mask_shape_mismatch')
    interior = cv2.erode((mask > 0).astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    valid = interior & np.isfinite(depth) & (depth > .03) & (depth < 4.)
    if valid.sum()<12:
        thin=(mask>0)&np.isfinite(depth)&(depth>.03)&(depth<4.)
        if thin.sum()>=24:
            valid=thin
    ys, xs = np.where(valid)
    if len(xs) < 12:
        raise ValueError('insufficient_valid_depth')
    z = depth[ys, xs]
    median = np.median(z)
    mad = np.median(np.abs(z-median))
    keep = np.abs(z-median) <= max(.015, 3.*mad)
    xs, ys, z = xs[keep], ys[keep], z[keep]
    if len(xs) < 12:
        raise ValueError('inconsistent_depth')
    K = np.asarray(calibration['K'], dtype=float)
    T = np.asarray(calibration['optical_to_world'], dtype=float)
    points = np.column_stack(((xs-K[0,2])*z/K[0,0], (ys-K[1,2])*z/K[1,1], z))
    world = points @ T[:3,:3].T + T[:3,3]
    position = np.median(world, axis=0)
    # Chromaticity is more stable across cameras than raw RGB brightness.
    pixels = rgb[ys, xs, ::-1].astype(float)
    chroma = pixels / np.maximum(pixels.sum(axis=1, keepdims=True), 1.)
    appearance = np.median(chroma, axis=0)
    covariance = np.cov(world[:,:2].T)
    values, vectors = np.linalg.eigh(covariance)
    yaw = 0.
    if values[1] > 1.3*max(values[0], 1e-8):
        short = vectors[:,0]
        yaw = math.atan2(short[1], short[0]) % math.pi
    # Rz(yaw) * Rx(pi), quaternion xyzw.
    quat = [math.cos(yaw/2), math.sin(yaw/2), 0., 0.]
    result = position.tolist(), appearance.tolist(), quat
    if return_details:
        extent = np.percentile(world[:,:2],95,axis=0)-np.percentile(world[:,:2],5,axis=0)
        return result + (extent.tolist(),)
    return result


def color_masks(rgb):
    """Diagnostic segmentation for colored cubes ONLY. Never called YOLO results."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    colors = [('red', ((hsv[:,:,0] < 12) | (hsv[:,:,0] > 170))),
              ('green', ((hsv[:,:,0] > 35) & (hsv[:,:,0] < 85))),
              ('blue', ((hsv[:,:,0] > 90) & (hsv[:,:,0] < 140)))]
    result = []
    for color, hue_mask in colors:
        mask = (hue_mask & (hsv[:,:,1] > 85) & (hsv[:,:,2] > 35)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, count):
            if 20 <= stats[i, cv2.CC_STAT_AREA] <= rgb.shape[0]*rgb.shape[1]*.1:
                result.append((color, (labels == i).astype(np.uint8), float(stats[i,cv2.CC_STAT_AREA])))
    return result


def reject_support_pixels(depth, mask, calibration):
    """Estimate the dominant horizontal tabletop from RGB-D, then remove its pixels.

    Shadows and an open cup's interior can otherwise dominate the surface median.
    This uses camera calibration and observed depth, never simulator object poses.
    """
    K=np.asarray(calibration['K'],dtype=float)
    T=np.asarray(calibration['optical_to_world'],dtype=float)
    ys,xs=np.where(np.isfinite(depth)&(depth>.03)&(depth<4.))
    z=depth[ys,xs]
    points=np.column_stack(((xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z))
    world=points@T[:3,:3].T+T[:3,3]
    region=((world[:,0]>.15)&(world[:,0]<.85)&(world[:,1]>-.55)&(world[:,1]<.45)&
            (world[:,2]>-.03)&(world[:,2]<.20))
    if region.sum()<100: raise ValueError('tabletop_not_observed')
    hist,edges=np.histogram(world[region,2],bins=np.arange(-.03,.202,.002))
    peak=int(np.argmax(hist))
    if hist[peak]<100: raise ValueError('tabletop_plane_not_supported')
    plane=(edges[peak]+edges[peak+1])/2
    output=np.zeros_like(mask,dtype=np.uint8)
    keep=(mask[ys,xs]>0)&(world[:,2]>plane+.006)
    output[ys[keep],xs[keep]]=1
    return output,float(plane)
