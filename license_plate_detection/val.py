from ultralytics import YOLO

def val(model):
    backbone = YOLO(model)
    results = backbone.val()
    