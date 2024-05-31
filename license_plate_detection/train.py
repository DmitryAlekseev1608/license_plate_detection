import torch
from ultralytics import YOLO

def train(path_data_config, model, epochs, imgsz, batch, DEVICE, workers):

    backbone = YOLO(model)
    results = backbone.train(
                    data=path_data_config,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    device=DEVICE,
                    workers=workers
                )
    success = backbone.export(imgsz=imgsz, format='torchscript', optimize=False, half=False, int8=False)