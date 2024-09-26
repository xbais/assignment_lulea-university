# ==================== #
#        CONFIG        #
# ==================== #

from headers import * # local_module
from losses import * # local_module

# BASIC PARAMS
#--------------------
python_version = '3.10.12'
system_requirements = ['cuda==11.8']
cpu_count = multiprocessing.cpu_count() 
poweroff_when_done = False
system_cooldown_time = 10 # seconds : wait for this time before exiting, this allows logger to complete logging

# PIPELINE PARAMS
#----------------
exp_name = 'v1' #'v6.5.1'
BASE_DIR = os.getcwd()
DATA_ROOT = os.path.join("../data")
EXP_DIR = os.path.join(BASE_DIR, 'experiments', exp_name)
max_processors = int(0.4*cpu_count) # Using a higher fraction of cpu_count can lead to CPU over-load and segmentation faults, recommended : <= 0.5
max_torch_multiprocessing_threads = 3 #max_processors
# Ensure that 'spawn' is used for starting processes

print(f'Max processors I will use = {max_processors} | Max Available = {cpu_count}')
#device = torch.device('cuda')
gpu_id = 0 
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f'Current device = {device}')

# DATASET PARAMS
#---------------
label_names = ['void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object', 'asphalt', 'build', 'log', 'person', 'fence', 'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']
original_label_ids = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 18, 19, 23, 27, 31, 33, 34]
colours = [[0,0,0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
label_mapping = {original_label_ids[_]:_ for _ in range(len(original_label_ids))}
num_classes = len(labels) 

#class_map = {0:"ground", 1:"vegetation", 2:"pole", 3:"car", 4:"truck", 5:"fence", 6:"powerline", 7:"building"}
class_map = {_:label_names[_] for _ in labels}
MAX_CORES = max_processors #10 # default = 10
dataloader_num_workers = 5 # 10
dataloader_timeout = 800 # seconds
dataloader_prefetch_factor = 5 # determines how many batches are preloaded in advance
shuffle_train_file_pts = True # Randomly shuffles train-file pts in every epoch while fetching data

# EARLY STOPPING PARAMS
#----------------------
es_patience = 9 #10
es_delta = 0.001 #0.005 # default 0

# ACTIVE LEARNING PARAMS
#-----------------------
#torch.set_default_device('cpu')
image_dims = [1920, 1200]
num_epochs = 15 #200 #200 # 30
train_batch_size = 1 # pimnet : (laptop) 4000 (pimdim=32,32, feats=3), (lab sys) 8000; vitb16 : 300 (lab system), 250 (laptop) (pimdim=32,32), 100 (pimdim=64,64)
val_batch_size = train_batch_size 
test_batch_size = train_batch_size 

# LEARNING RATE SCHEDULER PARAMS
#-------------------------------
initial_lr = 0.0001
lr_decay = initial_lr/10

test_image_path = 'test_image.png'