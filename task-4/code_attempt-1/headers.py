import os
from PIL import Image
import multiprocessing
import json
import numpy as np
from tqdm import tqdm

# PyTorch
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.utils.data

# Detectron2
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances


