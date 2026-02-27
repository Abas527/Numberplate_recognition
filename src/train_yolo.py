import yaml
from ultralytics import YOLO
import os
import torch

def load_configs(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs


def train():
    configs = load_configs("configs/config.yaml")
    model = YOLO(configs["model"])
    print("Starting training with following configurations:")
    for key, value in configs.items():
        print(f"  {key}: {value}")

    model.train(data=configs["data"], epochs=configs["epochs"], batch=configs["batch"], imgsz=configs["imgsz"],device=configs["device"], project=configs["project"], name=configs["name"])


    print("Model saved in the following directory:")
    print(os.path.join(configs["project"], configs["name"]))


def main():

    # print(torch.cuda.is_available())
    train()


if __name__ == "__main__":
    main()