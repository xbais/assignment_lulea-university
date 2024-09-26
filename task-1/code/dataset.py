# ========================= #
#   DATASET CLASS & UTILS   #
# ========================= #

from headers import * # Local_module
from utils import * # local_module
import utils
from config import * # local_module
import config # local_module

class RellisDataset(Dataset):
    def __init__(self):
        # Get train val test file names
        self.split_files = {'train':{'data':[], 'gt':[]}, 'val':{'data':[], 'gt':[]}, 'test':{'data':[], 'gt':[]}}
        self.split_files['train'], self.split_files['val'], self.split_files['test'] = self.get_file_paths()
        self.split, self.files, self.label_files = None, None, None
        return 
    
    def __len__(self):
        return len(self.files)

    def set_split(self, split:str):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.files = self.split_files[self.split]['data']
        self.label_files = self.split_files[self.split]['gt']

    def get_file_paths(self):
        # Load image split data
        with open(os.path.join(config.DATA_ROOT, 'Rellis_3D_image_split', 'train.lst'), 'r') as _:    
            data = [__.replace('\n', '') for __ in _.readlines()]
            train_data, train_gt = [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node', 'Rellis-3D', __.split(' ')[0]) for __ in data], [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node_label_id', 'Rellis-3D', __.split(' ')[1]) for __ in data]
            
        with open(os.path.join(config.DATA_ROOT, 'Rellis_3D_image_split', 'val.lst'), 'r') as _:    
            data = [__.replace('\n', '') for __ in _.readlines()]
            val_data, val_gt = [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node', 'Rellis-3D',__.split(' ')[0]) for __ in data], [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node_label_id', 'Rellis-3D',__.split(' ')[1]) for __ in data]

        with open(os.path.join(config.DATA_ROOT, 'Rellis_3D_image_split', 'test.lst'), 'r') as _:    
            data = [__.replace('\n', '') for __ in _.readlines()]
            test_data, test_gt = [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node', 'Rellis-3D',__.split(' ')[0]) for __ in data], [os.path.join(config.DATA_ROOT, 'Rellis_3D_pylon_camera_node_label_id', 'Rellis-3D',__.split(' ')[1]) for __ in data]

        return {'data':train_data, 'gt':train_gt}, {'data':val_data, 'gt':val_gt}, {'data':test_data, 'gt':test_gt}

    def get_unique_labels(self):
        _found_labels = []
        _colors = []
        while len(_found_labels) <= config.num_classes + 1:
            for file in tqdm(self.label_files):
                sequence, file_name = file.split('/')[-3], file.split('/')[-1]
                _file_path = os.path.join(file)
                _img = np.asarray(imageio.imread(_file_path))
                _unique_labels = list(np.unique(_img))

                # Check  if any labels are not present in _found_labels
                for label in _unique_labels:
                    if label not in _found_labels:
                        tqdm.write(f'Found new label : {label}')
                        _found_labels.append(label)

                        # Get the colours for this label
                        _colour_file_path = os.path.join('../data/Rellis_3D_pylon_camera_node_label_color/Rellis-3D/', sequence, 'pylon_camera_node_label_color', file_name)
                        _colour_img = np.asarray(imageio.imread(_colour_file_path))
                        colour = _colour_img[np.where(_img == label)][0]
                        tqdm.write(f'Colour found = {colour}')
                        _colors.append(colour)
        print('Done : ', _found_labels, _colors)
    
    def visualise(self, global_iter:int):
        """
        Visualise the PIM maps.
        """
        # randomly select 16 pim arrays and labels from the dataset
        file_id = 0
        pim_path = os.path.join(self.getitem_dir, str(file_id) + '_nn-data.h5_pim.h5')
        label_path = os.path.join(self.getitem_dir, str(file_id) + '_labels.h5')
        pim_array = utils.load_h5(file_path=pim_path)['data']
        labels_array = utils.load_h5(file_path=label_path)['data']
        idxs = np.random.choice(len(labels_array), 16)

        for channel_id in range(len(pim_array[0])):
            fig, axs = plt.subplots(4, 4, figsize=(15,15))
            images = [pim_array[idx][channel_id] for idx in idxs]
            labels = [int(labels_array[idx]) for idx in idxs]
            labels = [self.class_map[label] for label in labels]
            
            for i, ax in enumerate(axs.flatten()):
                if i < len(images):
                    ax.imshow(images[i])
                    ax.set_title(labels[i])
                else:
                    ax.remove()
            fig.suptitle("PIM Pt Count Visualisations")
            fig.tight_layout()
            #plt.show()
            plot_path = os.path.join(config.EXP_DIR, f'pim-vis_channel-{channel_id}_iter-{global_iter}.png')
            plt.savefig(plot_path)
            plt.clf() # Clear the plot
            plt.cla()

            logger.log(f'**Dataset channel-{channel_id} Visualisation**')
            logger.log(f'![Dataset channel-{channel_id} Visualisation]({plot_path.split("/")[-1]})')
            ############
        return
    
    def visualise_stats(self, exp_dir, global_iter):
        font_plt_title = {
            'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 20,
        }

        font_plt_text = {
            'family': 'serif',
            'color':  'black',
            #'weight': 'bold',
            'size': 15,
        }

        font_plt_text_small = {
            'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 7,
        }

        # Visualize class-wise statistics for the data
        items = np.int16(self.data['labels'])

        counts = Counter(items)
        class_counts = [counts[label] for label in labels]
        class_counts_total = np.sum(self.all_file_stats, axis=0).tolist()
        bar_width=0.4 # bar width
        # Set the positions of the bars
        r1 = np.arange(len(config.label_names))
        r2 = [x + bar_width for x in r1]

        # Create the figure and the first axis
        fig, ax1 = plt.subplots()
        ax2 = ax1.twiny()
        ax1.barh(r1, class_counts, height=bar_width, color='black')
        ax2.barh(r2, class_counts_total, height=bar_width, color='blue')
        ax1.xaxis.set_label_position('bottom')
        ax1.xaxis.tick_bottom()
        ax1.set_yticks(r1) #([r + bar_width / 2 for r in range(len(labels))], labels)
        ax1.set_yticklabels(labels, fontdict=font_plt_text)

        ax1.set_xlabel('Current Class counts', fontdict={'family':'serif', 'color':'black', 'weight':'bold', 'size':15})
        ax2.set_xlabel('Total Class counts', fontdict={'family':'serif', 'color':'blue', 'weight':'bold', 'size':15})
        ax1.set_ylabel('Classes', fontdict={'family':'serif', 'color':'black', 'weight':'bold', 'size':15})
        plt.title("Class Counts Visualisation", fontdict=font_plt_title)
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()

        # Adding text labels to each bar
        
        for i, value in enumerate(class_counts):
            # For black bars
            if value > max(class_counts) * 0.05:  # Adjust this threshold as needed
                ax1.text(value / 2, r1[i], str(value), color='white', ha='center', va='center', fontdict=font_plt_text_small)
            else:
                ax1.text(value + max(class_counts) * 0.01, r1[i], str(value), color='black', va='center', fontdict=font_plt_text_small)
            
            # For total stats bars
            value_total = class_counts_total[i]
            if value_total > max(class_counts_total) * 0.05:  # Adjust this threshold as needed
                ax2.text(value_total / 2, r2[i], str(value_total), color='white', ha='center', va='center', fontdict=font_plt_text_small)
            else:
                ax2.text(value_total + max(class_counts_total) * 0.01, r2[i], str(value_total), color='blue', va='center', fontdict=font_plt_text_small)
        
        #plt.xticks(fontfamily=font_plt_text['family'], fontsize=font_plt_text['size'])
        plt.yticks(labels, label_names, fontfamily=font_plt_text['family'], fontsize=font_plt_text['size'])

        #plt.show()
        plot_path = os.path.join(exp_dir, f'class-count_iter-{global_iter}_{self.split}.png')
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.savefig(plot_path)
        #plt.show()
        plt.clf() # Clear the plot
        plt.cla()
        logger.log('**Dataset Stats Visualisation**')
        logger.log(f'![Dataset Stats Visualisation]({plot_path.split("/")[-1]})')
        return
    
    def __getitem__(self, idx):
        data = torch.tensor(cv2.imread(self.files[idx])).permute(2, 0, 1)
        label_mask = np.array(imageio.imread(self.label_files[idx]))
        
        # Remap 
        label_mask = torch.tensor(utils.map_label(label_mask))
        
        #return torch.Tensor(data).float().permute(2, 0, 1), torch.Tensor(label_mask).float() 
        return data, label_mask
