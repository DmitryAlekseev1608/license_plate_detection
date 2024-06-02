from hydra import compose, initialize
from roboflow import Roboflow
import fire
import torch
import shutil

import license_plate_detection.train
import license_plate_detection.predict

def load_data_yolov8():

    rf = Roboflow(api_key="toA9WE6clS6ibaPq1YTQ")
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(4)
    dataset = version.download("yolov8")
    shutil.move('License-Plate-Recognition-4/test', 'data/')
    shutil.move('License-Plate-Recognition-4/train', 'data/')
    shutil.move('License-Plate-Recognition-4/valid', 'data/')
    shutil.move('License-Plate-Recognition-4/data.yaml', 'data/')
    shutil.rmtree('License-Plate-Recognition-4')

def train(model):

    license_plate_detection.train.train(path_data_config=COMMON.path_data_config,
                model=model,
                epochs=MODELS[model].epochs,
                imgsz=MODELS[model].imgsz,
                batch=MODELS[model].batch,
                DEVICE=DEVICE,
                workers=COMMON.worker)

def val(model):

    val.val(model=model)

def predict(source: str, model: str) -> None:

    license_plate_detection.predict.predict(
        source=source,
        model=model
    )

if __name__ == '__main__':
    
    initialize(version_base=None, config_path="configs", job_name="app")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = compose(config_name="models")
    COMMON = compose(config_name="common")
    fire.Fire()