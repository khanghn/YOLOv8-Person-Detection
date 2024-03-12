# Convert crowhuman annotations to COCO format
python crowdhuman_to_coco.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/

# Convert crowhuman annotations COCO format to YOLO format
python crowhuman_to_yolo.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/ \
                                --split val train --dataset crowdhuman -np 0 

#Convert COCO annotations to YOLO format
python coco_to_yolo.py 

# # Combine dataset
# python mix_coco_crow.py --paths /home/nguyenkhang/Documents/YOLOv8/datasets --out_path /home/nguyenkhang/Documents/YOLOv8/datasets/coco_crow
