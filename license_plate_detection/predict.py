from ultralytics import YOLO

def predict(model, source):

    model = YOLO(model)
    results = model(source, save=True)