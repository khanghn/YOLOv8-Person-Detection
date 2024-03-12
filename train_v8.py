from yolo.engine.model import YOLO
from yolo.utils import DEFAULT_CFG
import torch 
# Load a model
def train(cfg=DEFAULT_CFG, use_python=True):
    
    data = cfg.data   
    device = cfg.device if cfg.device is not None else ''
    args = dict(model=cfg.pretrained_wts, data=data, device=device)
    if use_python:
        from yolo.engine.model  import YOLO
        model=YOLO(cfg.model)  
        model.train(**args)
  

if __name__ == "__main__":
    train()
