from engine import YOLO


# model = YOLO('yolov8m.yaml')
model = YOLO('/kaggle/input/bestt/pytorch/model/1/best (1).pt')
model.train(data='coco-crow.yaml', resume=True, epochs=50, batch=-1, imgsz=640)