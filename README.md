# YOLOv8-Person-Detection

## 1. Installation

```bash
git clone https://github.com/khanghn/YOLOv8-Person-Detection.git  # clone
cd YOLOv8-Person-Detection
pip install -r requirements.txt  # install
```

## 2. Datasets

#### Convert crowhuman annotations to COCO format
```bash
python crowdhuman_to_coco.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/
```

#### Convert crowhuman annotations COCO format to YOLO format
```bash
python crowhuman_to_yolo.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/ \
                                --split val train --dataset crowdhuman -np 0 
```

#### Convert COCO annotations to YOLO format
```bash
python coco_to_yolo.py 
```

#### Combine dataset
```bash
python mix_coco_crow.py --paths /home/nguyenkhang/Documents/YOLOv8/datasets --out_path /home/nguyenkhang/Documents/YOLOv8/datasets/coco_crow
```

## 3. Training

### 3.1 Using official models
`pip install ultralytics `

Run offical command 
```bash
yolo task=detect    mode=train   data=<data.yaml path>      model=yolov8n.pt        args...
          classify       predict        coco-128.yaml       yolov8n-cls.yaml  args...
          segment        val                                yolov8n-seg.yaml  args...
                         export                             yolov8n.pt        format=onnx  args...
```
### 3.2 Custom models
Setup the model configs in `yolo/cfg/default.yaml` and dataset configs in `data/coco_crow.yaml`

`python train_v8.py`