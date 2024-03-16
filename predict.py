from engine import YOLO
from engine.utils import ASSETS
from engine.models.yolo.detect import DetectionPredictor


# model = YOLO('yolov8m.yaml')
# model.predict(data='coco-crow.yaml', epochs=10, batch=-1, imgsz=640)


args = dict(model='/home/nguyenkhang/Downloads/best (1).pt', source='/home/nguyenkhang/Documents/YOLOv8/datasets/coco_crow/images/val')
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()