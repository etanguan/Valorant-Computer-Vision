import torch
import os
from IPython.display import Image, clear_output  # to display images
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")

os.environ["DATASET_DIRECTORY"] = "/content/datasets"


rf = Roboflow(api_key="yourKey")
project = rf.workspace("david-hong").project("valorant-enemy")
dataset = project.version(2).download("yolov5")

#python train.py --img 416 --batch 16 --epochs 300 --data=C:\content\datasets\valorant-enemy-2\data.yaml --weights yolov5s.pt --cache
