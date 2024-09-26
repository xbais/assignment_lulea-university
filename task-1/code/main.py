#!/DATA/phd-interview_stage-2/task-1_image-seg/code/pyvenv_3.10.12/bin/python

# ==================== #
#          MAIN        #
# ==================== #

# BY Aakash Singh Bais, 2024

import sys, os, subprocess, shutil
sys.path.append('/media/aakash/active/_xosuit/py_utils/')
from autodep import install
install(script_path=__file__, python_version='3.10.12', sys_reqs=['cuda==11.8'])

# Import All modules
from headers import * # local_module
import networks # local_module
import utils # local_module
from config import * # local_module
from dataset import * # local_module
import train_val # local_module
import config # local_module

exp_dir = 'experiments'
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)
exp_list = os.listdir(exp_dir)
print(f'Existing experiments:\n' + str(exp_list))

if exp_name in exp_list:
    print(f'🟠 [black on yellow][italic] WARNING [white on black]: [yellow on black] Experiment directory already exists, will be over-written')
    shutil.rmtree(exp_dir)

os.makedirs(config.EXP_DIR)
os.mkdir(os.path.join(config.EXP_DIR, 'val_cm'))
#os.mkdir(os.path.join(config.EXP_DIR, 'test_cm'))


logger.init()
#atexit.register(logger.set_error)
# Set up signal handlers for SIGINT and SIGTERM
signal.signal(signal.SIGINT, logger.log_exit)
signal.signal(signal.SIGTERM, logger.log_exit)

logger.log(f'# Task-1 (Image Segmentation) : {config.exp_name}')
#logger.log(f'![logo]({os.path.join(config.BASE_DIR, "logo.jpg")})')

error = ""

try:
    model = networks.UNet()  # 3 input channels (RGB), 1 output channel (segmentation mask)
    print(model)
    dataloaders = train_val.get_loaders()
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
    trained_model = train_val.train(num_epochs=config.num_epochs, model=model, optimizer=optimizer, loaders=dataloaders)
    train_val.test()
except KeyboardInterrupt:
    error = "KeyboardInterrupt"
    logger.set_error(error)
    sys.exit(1)
except Exception as e:
    error = e
    print(error)
    print(traceback.format_exc())
    logger.set_traceback(traceback.format_exc())
    logger.set_error(error)
    pdb.pm() # Break into PDB debugger in post-mortem mode when an exception happens
    sys.exit(0)

logger.set_error(error="")