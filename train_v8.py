from ultra import YOLO


model = YOLO('yolov8m.yaml')
model.train(data='coco-crow.yaml', epochs=100, batch=-1, imgsz=640)