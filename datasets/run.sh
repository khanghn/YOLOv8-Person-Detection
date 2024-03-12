python crowdhuman_to_coco.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/

python coco_to_yolo.py --data_path /home/nguyenkhang/Documents/YOLOv8/datasets/crowdhuman/ \
                                --split val train --dataset crowdhuman -np 0 

# # Combine dataset
# python mix_mot_ch.py --paths /home/nguyenkhang/Documents/YOLOv8/datasets --out_path /home/nguyenkhang/Documents/YOLOv8/datasets/coco_crow
