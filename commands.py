from hydra import compose, initialize
import fire
import torch

import license_plate_detection.train as train
import license_plate_detection.val as val

def train(model):

    train.train(path_data_config=PATH_DATA_CONFIG,
                model=model,
                epochs=MODELS[model].epochs,
                imgsz=MODELS[model].imgsz,
                batch=MODELS[model].batch,
                DEVICE=DEVICE,
                workers=WORKERS)
    
def val(model):

    val.val(model=model)

if __name__ == '__main__':
    
    PATH_DATA_CONFIG = '../configs/data.yaml'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS = compose(config_name="models")
    WORKERS = 8

    initialize(version_base=None, config_path="configs", job_name="app")
    fire.Fire()