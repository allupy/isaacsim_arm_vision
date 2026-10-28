import cv2,numpy as np

def locate_support(rgb,depth,calibration,identifier):
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),cv2.aruco.DetectorParameters())
    corners,ids,_=detector.detectMarkers(rgb)
    if ids is None or identifier not in ids.flatten():raise ValueError('destination_marker_not_visible')
    matches=[i for i,v in enumerate(ids.flatten()) if int(v)==identifier]
    if len(matches)!=1:raise ValueError('ambiguous_destination_marker')
    quad=corners[matches[0]][0];center=quad.mean(0);quad=center+(quad-center)*.55
    mask=np.zeros(depth.shape,np.uint8);cv2.fillConvexPoly(mask,quad.astype(np.int32),1)
    ys,xs=np.where((mask>0)&np.isfinite(depth)&(depth>.03)&(depth<4.))
    if len(xs)<16:raise ValueError('destination_depth_missing')
    z=depth[ys,xs];K=np.asarray(calibration['K']);T=np.asarray(calibration['optical_to_world'])
    points=np.column_stack(((xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z))@T[:3,:3].T+T[:3,3]
    position=np.median(points,axis=0)
    if np.ptp(np.percentile(points[:,2],[10,90]))>.008:raise ValueError('destination_not_horizontal_or_occupied')
    if not (.2<position[0]<.8 and -.5<position[1]<.4 and .01<position[2]<.15):raise ValueError('destination_outside_workspace')
    return position.tolist(),corners[matches[0]][0].tolist()
