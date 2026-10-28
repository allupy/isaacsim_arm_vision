from ultralytics import YOLOWorld
model=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
image='/experiment/runs/yolo_cup_v3_20261028_1/001_wrist.png'
for word in ['cup','mug','coffee mug','red mug','red cup']:
    model=YOLOWorld('/ros2_ws/isaacsim_arm_vision/yolov8s-worldv2.pt')
    classes=[word,'banana','apple'];model.set_classes(classes)
    r=model.predict(image,imgsz=1280,conf=.005,max_det=8,verbose=False)[0]
    print(word,[(classes[int(b.cls)],round(float(b.conf),3),[round(x) for x in b.xyxy.tolist()[0]]) for b in r.boxes],flush=True)
