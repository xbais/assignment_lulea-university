#!/DATA/phd-interview_stage-2/task-4_thermal-object-detection/code/pyvenv_3.10.12/bin/python
import sys, os, subprocess, shutil
sys.path.append('/media/aakash/active/_xosuit/py_utils/')
from mlmaid import install # local_module
install(script_path=__file__, python_version='3.10.12', sys_reqs=['cuda==11.8'])

from headers import * # local_module
import config # local_module

# Register the COCO-like dataset (your own dataset)
register_coco_instances("custom_train", {}, config.train_data_annotations, config.train_data_path)
register_coco_instances("custom_val", {}, config.val_data_annotations, config.val_data_path)

class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        self.cfg = cfg
        self.val_loss = []

    def after_epoch(self):
        # Evaluate model and calculate validation loss
        evaluator = COCOEvaluator("custom_val", self.cfg, False, output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "custom_val")
        val_metrics = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        
        # Store validation loss (you can also store other metrics like mIoU here)
        print(f"==> Validation metrics: {val_metrics}")
        
        # Save model checkpoint at every epoch
        torch.save(self.trainer.model.state_dict(), f"./model_epoch_{self.trainer.iter}.pth")
        print(f"==> Model checkpoint saved at epoch {self.trainer.iter}")

def train_model():
    # Register your custom dataset
    #register_coco_instances("custom_train", {}, "path/to/annotations_train.json", "path/to/train_images")
    #register_coco_instances("custom_val", {}, "path/to/annotations_val.json", "path/to/val_images")
    
    # Setup configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)
    cfg.DATALOADER.NUM_WORKERS = config.dataloader_num_workers
    cfg.MODEL.WEIGHTS = ""  # Train from scratch
    cfg.SOLVER.IMS_PER_BATCH = config.images_per_batch
    cfg.SOLVER.BASE_LR = config.base_lr
    cfg.SOLVER.MAX_ITER = config.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.num_classes  # Update with number of classes in your dataset
    cfg.OUTPUT_DIR = "./output"  # Directory to save the trained models and logs

    # Initialize the trainer
    trainer = DefaultTrainer(cfg)

    # Add a hook for validation after every epoch
    val_loss_hook = ValidationLossHook(cfg)
    trainer.register_hooks([val_loss_hook])

    # Start training
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    train_model()