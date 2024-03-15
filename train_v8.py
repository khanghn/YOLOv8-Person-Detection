from engine import YOLO


model = YOLO('yolov8m.yaml')
model.train(data='coco-crow.yaml', epochs=10, batch=32, imgsz=640, device=[0,1])