# ==================== #
#        CONFIG        #
# ==================== #

from headers import * # local_module

# BASIC PARAMS
#--------------------
python_version = '3.10.12'
system_requirements = ['cuda==11.8']
cpu_count = multiprocessing.cpu_count() 

# PIPELINE PARAMS
#----------------
data_root = '../data/FLIR_ADAS_v2/'
train_data_path, val_data_path, test_data_path = os.path.join(data_root, 'images_thermal_train'), os.path.join(data_root, 'images_thermal_val'), os.path.join(data_root, 'video_thermal_test')
train_data_annotations, val_data_annotations, test_data_annotations = os.path.join(train_data_path, 'coco.json'), os.path.join(val_data_path, 'coco.json'), os.path.join(test_data_path, 'coco.json')
train_data_images, val_data_images, test_data_images = os.path.join(train_data_path, 'data'), os.path.join(val_data_path, 'data'), os.path.join(test_data_path, 'data')
train_size, val_size, test_size = len(os.listdir(train_data_images)), len(os.listdir(val_data_images)), len(os.listdir(test_data_images))

# DATASET PARAMS
#---------------
label_names = ['person', 'bike', 'car', 'motor', 'bus', 'train', 'truck', 'light', 'hydrant', 'sign', 'dog', 'skateboard', 'stroller', 'scooter', 'other_vehicle', 'concrete', 'barrier', 'puddle', 'mud', 'rubble']
original_label_ids = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 17, 37, 73, 77, 79]
#colours = [[0,0,0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153], [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64], [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204], [170, 170, 170], [41, 121, 255], [134, 255, 239], [99, 66, 34], [110, 22, 138]]
num_classes = len(label_names) 
labels = list(np.arange(num_classes))
#label_mapping = {original_label_ids[_]:_ for _ in labels}


#class_map = {0:"ground", 1:"vegetation", 2:"pole", 3:"car", 4:"truck", 5:"fence", 6:"powerline", 7:"building"}
class_map = {_:label_names[_] for _ in labels}
MAX_CORES = cpu_count
dataloader_num_workers = 2

# EARLY STOPPING PARAMS
#----------------------
#es_patience = 9 #10
#es_delta = 0.001 #0.005 # default 0

# ACTIVE LEARNING PARAMS
#-----------------------
#torch.set_default_device('cpu')
num_epochs = 5 #200 #200 # 30
images_per_batch = 2
batch_size_per_image = 1 # 128 (Goes out of memory at higher values)
num_iters_per_epoch = int(train_size / images_per_batch)
max_iter = num_iters_per_epoch*num_epochs #30000
base_lr = 0.00025

# Pre-trained Model To Load for testing
pretrained_model_for_testing = 'output/model_0024999.pth'