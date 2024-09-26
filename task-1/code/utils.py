# ==================== #
#       UTILITIES      #
# ==================== #

from headers import * # local_module
from config import * # local_module
import config # local_module

#lock = multiprocessing.Lock()

def run_in_background(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, func, *args, **kwargs)
    return wrapper

def check_network_shared_resources(model, model_copy):
    for param1, param2 in zip(model.parameters(), model_copy.parameters()):
        assert param1.data_ptr() != param2.data_ptr(), "Parameters are still shared!"

    for buffer1, buffer2 in zip(model.buffers(), model_copy.buffers()):
        assert buffer1.data_ptr() != buffer2.data_ptr(), "Buffers are still shared!"

def shuffle_tensor(tensor:torch.tensor) -> torch.tensor:
    '''
    Shuffles the elements of tensor along first dim, works similar to np.random.shuffle
    '''
    return tensor[torch.randperm(tensor.shape[0])]

def batchify_tensor(tensor:torch.tensor, batch_size:int, device:str='cpu')->list:
    return [tensor[i:i + batch_size].cpu() for i in range(0, tensor.size(0), batch_size)] if device == 'cpu' else [tensor[i:i + batch_size].cuda() for i in range(0, tensor.size(0), batch_size)]

def print_md(md_string:str) -> None:
    md = Markdown(md_string)
    console.print(md)
    return

def sequential_process(func, iterable, show_progress=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if show_progress:
            # Use rich's track for a progress bar
            result = [func(*item) if isinstance(item, tuple) else func(item)
                      for item in track(iterable)]
        else:
            # Most efficient alternative for iteration
            result = [func(*item) if isinstance(item, tuple) else func(item)
                      for item in iterable]
        return result
    return wrapper

def parallel_process(func, iterable, show_progress=False):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pool = multiprocessing.Pool(processes=cpu_count())

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Processing...", total=len(iterable))
                
                result = []
                for res in pool.imap(func, iterable):
                    result.append(res)
                    progress.advance(task)
        else:
            result = pool.map(func, iterable)
        
        pool.close()
        pool.join()
        return result
    return wrapper

def run_periodically_async(interval, func, *args, **kwargs):
    """
    Run a function periodically in a separate thread.
    
    :param interval: Time in seconds between each function call.
    :param func: Function to run periodically.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    """
    def periodic_wrapper():
        while True:
            func(*args, **kwargs)
            time.sleep(interval)

    thread = threading.Thread(target=periodic_wrapper)
    thread.daemon = True
    thread.start()

class Logger():
    '''
    Way this will work:
    1. 
    '''
    @typechecked
    def __init__(self, evaluation_mode:bool=False) -> None:
        self.evaluation_mode = evaluation_mode
        
        self.error = ""
        self.traceback = ""
        
        self.times = nx.DiGraph() # This will store the times
        self.times.add_node(0, data={'name':'Total Time', 'time':datetime.datetime.now()}) # Root node of the graph
        self.times_ids_data = [] # This will keep track of ids of nodes added in graph : [node_level(1-6), node_name]
        self.last_parent_names = [None]*6 # This will store the names of the last headings seen
        #self.current_parent_id = None
        self.last_node = None
        self.graph_node_color_dict = None
        self.log_file_name = None
        
    @typechecked
    def _profiler_get_level(self, node_name:str):
        return list(node_name.strip().split(' ')[0]).count('#') if '#' in node_name else None
    
    #@typechecked   
    #def _profiler_node_exists(self, level:int, node_name:str) -> int: # should have an additional argument to check the last element of self.times_ids_data first
    #    return self.times_ids_data.index([level, node_name]) if [level, node_name] in self.times_ids_data else -1

    @typechecked
    def _profiler_append_node(self, name, time=0):
        '''
        This function appends a time object to the profiler
        '''
        
        level = self._profiler_get_level(name)
        node = {'name': name, 'time':time}
        node_id_data = [level, name]
        node_id = len(self.times_ids_data)+1
        if not level: # If the logstring doesn't have #, and hence is not a heading
            return

        def get_parent():
            '''
            This returns the parent node for a given node...
            '''
            parent_id = None
            if level > self.times_ids_data[-1][0]:
                #print('Level more than the last element')
                #parent = self.times_ids_data[-1]
                parent_id = len(self.times_ids_data) - 1
            elif level <= self.times_ids_data[-1][0]:
                #print('Level less than the last element')
                _ = len(self.times_ids_data) - 1
                while _ > -1:#for id in range(len(self.times_ids_data)):#self.times_ids_data[::-1]: # Check last element first
                    element = self.times_ids_data[_]
                    if element[0] < level:
                        #parent = element
                        parent_id = _
                        break
                    _ -= 1
                if not parent_id:
                    print(f'Cannot find suitable parent. Level = {level}, Times IDS Data = {self.times_ids_data}')
            else:
                print(f'Weird condition encountered : self.times_data = {self.times_ids_data}, level = {level} \n Cannot find a parent, although it should exist.')
            return parent_id
                    

        # Add the new node
        if not self.times.nodes[0]:    
            self.times.add_node(0, data={'name':'Total Time', 'time':datetime.datetime.now()})
            print(f'New root added...')
        self.times.add_node(node_id, data=node) # add a node with a unique id and data
        if level == 1:
            self.times.add_edge(0, node_id) # Make it a direct child of the root node
        elif level > 1: # Add an edge between the child and the parent
            #print(f'Node = {node}')
            parent_id = get_parent()
            # Check if the parent exists
            #assert self._profiler_node_exists(parent_id_data[0], parent_id_data[1]) != -1, f"Parent node doesn't exist. Node id = {node_id_data}, \n Current nodes : {self.times_ids_data}"
            assert isinstance(parent_id, int), f"Parent node doesn't exist. Node id = {node_id_data}, \n Current nodes : {self.times_ids_data}"
            # Create a directed edge from the parent to the child
            self.times.add_edge(parent_id+1, node_id)
        else:
            print(f'Unknown condition triggered at level : {level}, data : {node}')
        self.times_ids_data.append(node_id_data)
        # Generate a new profiler graph
        self.graph_node_color_dict = profiler_graph(self.times, save_path=os.path.join(config.EXP_DIR, 'profiler_graph.html'), color_dict=self.graph_node_color_dict if self.graph_node_color_dict else None)
        return
    
    def eval_mode(self):
        self.evaluation_mode = True
        
    def init(self):
          
        self.log_file_path = os.path.join(config.BASE_DIR, 'experiments', config.exp_name)
        self.log_file_name = 'evaluation_logs.html' if self.evaluation_mode else 'logs.html'
        print_md(f'**NOTE :** Log file location : file://{os.path.join(self.log_file_path, self.log_file_name)}')

        # Copy the CSS and JS to experiment directory
        for file_name in ['logs.css', 'logs.js']:
            shutil.copyfile(os.path.join(config.BASE_DIR, file_name), os.path.join(config.EXP_DIR, file_name))
        if not os.path.isdir(self.log_file_path) : 
            print(f'Not a dir : {self.log_file_path}')
            exit()

        # Initialise the HTML
        with open(os.path.join(self.log_file_path, self.log_file_name), 'w') as _:
            _.writelines(
                [
                    '<html>\n',
                    f'<head><link rel="stylesheet" href="logs.css"/><script type="text/javascript" src="logs.js"></script></head>\n',
                    '<body>\n',
                    '<div id="progressbar"></div>\n',
                 ]
                )
        # Initialise start time
        self.start_time = datetime.datetime.now()

        # Initialise the buffer directory 
        #   Every log command will write a uniquely named separate text file to the buffer (to prevent simultaneous logging during multiprocessing)
        #   These log chunks will be added to the main log file at regular intervals according to the time at which the file was created
        #   Once resolved into the main log, the chunks will be deleted
        self.buffer_path = os.path.join(self.log_file_path, '.log.buffer')
        if not os.path.isdir(self.buffer_path):
            print('Log buffer directory does NOT exist. Creating.')
            os.mkdir(self.buffer_path)

        # Resolve log buffer periodically
        run_periodically_async(3, self.resolve_buffer)

    def resolve_buffer(self):
        if os.path.isdir(self.buffer_path):
            # List all chunk files in the buffer : sorted
            log_chunk_list = sorted(os.listdir(self.buffer_path))
            log_chunk_list = [os.path.join(self.buffer_path, file) for file in log_chunk_list]
            
            def read_chunk(path:str):
                with open(path, 'r') as __:
                    data = __.read()
                return data
            
            log_buffer_data = [read_chunk(chunk_path) for chunk_path in log_chunk_list]

            # Write to main logs
            with open(os.path.join(self.log_file_path, self.log_file_name), 'a') as _:
                _.writelines(log_buffer_data)
            
            # Delete chunk files
            for file in log_chunk_list:
                os.remove(file)
        
        return

    def bypass_log(function):
        return print

    #@bypass_log
    def log(self, log_string:str):
        """
        This is the function that does all the logging, this must be used everywhere for logging.
        Tasks:
        1. Print the log (markdown line) to the terminal
        2. Append the log to the log file (markdown)
        3. Re-Generate the renderable HTML file of the log
        """
        # Print to terminal
        print_md(log_string)

        # Add corresponding HTML version logged md using Marko 
        # Marko Documentation : https://marko-py.readthedocs.io/en/latest/index.html
        # Extract hours, minutes, and seconds
        time_difference = datetime.datetime.now() - self.start_time 
        total_seconds = time_difference.total_seconds()
        hours = f'{int(total_seconds // 3600)} h' if int(total_seconds // 3600) > 0 else ''
        minutes = f'{int((total_seconds % 3600) // 60)} m' if int((total_seconds % 3600) // 60) > 0 else ''
        seconds = f'{int(total_seconds % 60)} s' if int(total_seconds % 60) > 0 else ''
        time_tag = '<div class="timetag">+ ' + f'{hours} {minutes} {seconds}'.strip() + '</div>'
        
        # Add to graph
        #current_level = self._profiler_get_level(log_string)
        self._profiler_append_node(name=log_string, time=datetime.datetime.now())

        with open(os.path.join(self.buffer_path, str(time.time())), 'a') as _:
            _.writelines([f'<div class="line" title="{str(datetime.datetime.now(pytz.timezone("Asia/Kolkata")))}">{time_tag}' + marko.convert(log_string) + "</div>\n"])

    def set_error(self, error:str=None):
        #self.resolve_buffer()
        self.error = error
        self.traceback = str(self.traceback).replace("\n","<br>")
        if error != None:
            if self.error:    
                with open(os.path.join(self.log_file_path, self.log_file_name), 'a') as _:
                    _.writelines([
                        f'<div id="error"><h2>Terminated Due To Error</h2><b>{self.error}</b><hr>{self.traceback}</div>'
                    ])
            else:
                with open(os.path.join(self.log_file_path, self.log_file_name), 'a') as _:
                    _.writelines([
                        f'<div id="finish"><h2>Program Execution Finished Successfully!</h2></div>'
                    ])
            return error
    
    def set_traceback(self, trace:str):
        self.traceback = trace
        return self.traceback
    
    def log_exit(self, signal, frame):
        """
        This function should be called in case the program terminates
        """
        print(self.error)
        print(self.traceback)
        self.set_error()
        time.sleep(config.system_cooldown_time) # Sleep for 5 seconds and let logs resolve
        if config.poweroff_when_done:
            os.system('poweroff')
        else:
            sys.exit(0)

logger = Logger()

# Example usage
#spherical = convert_cartesian_to_spherical(1.0, 1.0, 1.0)
#spherical = ctypes.cast(spherical, ctypes.POINTER(SphericalCoordinates)).contents

def histogram_of_unique_items(input_list:list):
    # Use Counter to count the occurrences of each unique item
    counter = Counter(input_list)
    
    # Convert the counter to a dictionary for easier use
    histogram = dict(counter)
    
    return histogram

def save_to_h5(file_path:str, dataset_names:list=[], arrays:list=[], overwrite:bool=False) -> None:
    """
    Save multiple numpy arrays to a single HDF5 file.

    Parameters:
    file_path (str): Path to the HDF5 file.
    dataset_names (list of str): List of dataset names. If dataset names are not provided at all, then 'data' is used
    arrays (list of np.ndarray): List of numpy arrays.
    """
    mode = 'a'
    if overwrite and os.path.isfile(file_path):
        mode = 'w'
        os.remove(file_path)
    elif not os.path.isfile(file_path):
        mode = 'w'
        
    if dataset_names == []:
        if len(arrays) > 1:
            raise Exception('No dataset names provided, dataset naming will default to "data". But, number of arrays are more than. Cannot be stored in one dataset.')
        dataset_names = ['data' for _ in range(len(arrays))]
    if len(dataset_names) != len(arrays):
        raise ValueError("The number of dataset names must match the number of arrays.")

    # Open the HDF5 file
    with h5py.File(file_path, mode) as f:
        for name, data in zip(dataset_names, arrays):
            if name in f:
                # Dataset exists, expand it
                dset = f[name]
                current_shape = dset.shape
                new_shape = (current_shape[0] + data.shape[0],) + current_shape[1:]
                dset.resize(new_shape)
                dset[current_shape[0]:] = data
                #if 'train/payload/0_nn-data.h5' in file_path and name == 'distances':    
                #    print(Fore.RED + f'#### Prev shape = {dset.shape[0]} | New shape = {new_shape[0]}' + Style.RESET_ALL)
            else:
                # Create a new dataset
                max_shape = (None,) + data.shape[1:]
                f.create_dataset(name, data=data, maxshape=max_shape, compression=H5_COMPRESSION_LEVEL)
                #if 'train/payload/0_nn-data.h5' in file_path and name == 'distances': 
                #    print(Fore.RED + f'#### Prev shape = {0} | New shape = {data.shape[0]}' + Style.RESET_ALL)

@run_in_background
def save_to_h5_async(file_path:str, dataset_names:list=[], arrays:list=[], overwrite:bool=False) -> None:
    save_to_h5(file_path=file_path, dataset_names=dataset_names, arrays=arrays, overwrite=overwrite)

def load_h5(file_path:str):
    """
    This function loads a given h5 file and returns a dictionary of all the datasets found within it by name : value pairs

    Args:
        file_path : the location of the h5 file to load

    Returns:
        datasets : a dictionary with dataset names as keys and dataset data as values
    """
    with h5py.File(file_path, 'r') as f:
        datasets = {}
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                datasets[key] = np.array(f[key])
    return datasets

def load_large_csv_to_numpy(csv_file, chunk_size=1000):
    # Count total number of rows in the CSV file
    total_rows = sum(1 for _ in open(csv_file))

    # Initialize progress bar
    progress_bar = tqdm(total=total_rows, desc=f"Loading {csv_file}")

    chunks = []
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        chunks.append(chunk.to_numpy())
        progress_bar.update(len(chunk))

    progress_bar.close()

    return np.concatenate(chunks, axis=0)

def dataset_visualisation(pim_dataset):
    logger.log('# Dataset Visualisations and Statistics')
    # FOR VISUALISATION OF THE PIM MAPS, USE THE FOLLOWING
    pim_dataset.visualise()
    
    # Visualize class-wise statistics for the data
    items = np.int16(pim_dataset.data['labels'])
    
    counts = Counter(items)
    class_counts = [counts[label] for label in labels]
    categories = labels
    #plt.bar(labels, class_counts)
    fig, ax = plt.subplots()
    bars = ax.barh(labels, class_counts, color='black')
    
    # Define a threshold to decide when to place the label inside or outside
    threshold = 3000
    
    # Add numeric labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = 0.5*width if width > threshold else width*1.5  # Position inside if bar is long enough, otherwise just outside
        color = 'white' if width > threshold else 'black'  # Font color
        alignment = 'center' if width > threshold else 'left'
        
        ax.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f'{int(width)}',  # Convert width to an integer for display
            ha=alignment,  # Horizontal alignment
            va='center',   # Vertical alignment
            color=color,
            fontsize=10
        )
        
    plt.xlabel('Classes')  
    plt.ylabel('Class counts')
    plt.title("Class Distribution")
    plt.yticks(labels, label_names, rotation='horizontal')
    plt.show()  
    print(f"Length of the dataset = {str(len(pim_dataset))}")

# Training Utils

def get_iou_stats(preds:np.ndarray, gt:np.ndarray, ignore_last_class:bool=False):
    cm = confusion_matrix(preds, gt, labels=config.labels[:-1]) if ignore_last_class else confusion_matrix(preds, gt, labels=config.labels)
    
    # Number of classes
    num_classes = len(cm)
    class_ious = []
    
    # Initialize variables to store TP, FP, FN, and TN for each class
    TP = [0] * num_classes
    FP = [0] * num_classes
    FN = [0] * num_classes
    TN = [0] * num_classes
    
    # Calculate TP, FP, FN, TN for each class
    for i in range(num_classes):
        TP[i] = cm[i, i]
        FP[i] = sum(cm[:, i]) - TP[i]
        FN[i] = sum(cm[i, :]) - TP[i]
        TN[i] = sum(np.diag(cm)) - TP[i]
    
    # Print results for each class
    #for i in range(num_classes):
        #print(f"Class {i}:")
        #print(f"TP: {TP[i]}, FP: {FP[i]}, FN: {FN[i]}, TN: {TN[i]}")
        #class_iou = TP[i]/(FP[i]+TP[i]+FN[i])
        #class_ious.append(class_iou)
    #    pass
    return np.array(TP), np.array(TN), np.array(FP), np.array(FN), cm

# Softmax wrapper
def softmax(logits:torch.tensor):
    return F.softmax(logits, dim=1) # dim = dim / axis along which softmax is to be calculated

def logits2confidence(logits):
    """
    This function converts the logits to confidence scores.
    """
    #print(logits.shape)
    #logits = logits.cpu().detach().numpy()
    probs = softmax(logits)
    confidence_scores = torch.max(probs, dim=1).values
    return confidence_scores.tolist()

def logits2preds(logits):
    """
    This function converts the logits to predictions.
    Trick from : https://annyme.medium.com/a-beginners-guide-to-numpy-apply-along-axis-lambda-133db14837d3
    """
    #if ignore_lsat_label:
    #    logits = logits[:,:-1]
    #print(logits.shape)
    #logits = logits.cpu().detach() #.numpy()
    logits = logits.float()
    #print(logits[0])
    #print(f"logits shape = {logits.shape}")
    #preds = np.apply_along_axis(sigmoid, 1, logits)
    probs = softmax(logits) #np.apply_along_axis(softmax, 1, logits)
    preds = torch.argmax(probs, dim=1)
    #print("Predictions = ", preds.dtype, preds)
    #print(f'Predictions shape = {preds.shape}')
    #print(f'Preds[0] = {preds[0]}')
    return preds

def plotPimNetTrainingStats(stats:dict, exp_dir:str, global_iter:int, mode:str='train') -> None:
    if mode == 'train':
        logger.log("**Plotting Training Stats...**")
        epochs = np.arange(len(stats[list(stats.keys())[0]]))

        # Plotting the first series
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('% Accuracies / mIoU', color=color)
        ax1.plot(epochs, stats['val_accs'], marker=',', linestyle=":", color=color, label='% Val Acc')
        ax1.plot(epochs, stats['train_mious'], marker=',', linestyle="-", color="tab:cyan", label='% Train mIoU')
        ax1.plot(epochs, stats['val_mious'], marker=',', linestyle="-", color="tab:blue", label='% Val mIoU')
        #ax1.plot(epochs, per_epoch_stats['test_mious'], marker=',', linestyle="-", color="tab:green", label='% Test mIoU')

        #ax1.plot(epochs, per_epoch_stats['mious'], marker='o', color='tab:green', label='mIoU')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0, top=100)

        # Creating a secondary y-axis for the second series
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Losses', color=color)
        ax2.plot(epochs, stats['val_losses'], marker=',', color=color, label='Normalised Val Loss')
        #ax2.plot(epochs, per_epoch_stats['test_losses'], marker=',', linestyle=":", color="tab:pink", label='Test Losses')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)

        # Third axis for learning rate
        ax3 = ax1.twinx()
        color = 'tab:green'
        ax3.set_ylabel('Learning Rate', color=color)
        ax3.plot(epochs, stats['lr'], marker=',', color=color, label='Learning Rate')
        #ax2.plot(epochs, per_epoch_stats['test_losses'], marker=',', linestyle=":", color="tab:pink", label='Test Losses')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim(bottom=0)

        # To prevent the two y-axes from overlapping
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.spines['right'].set_color('green')

        # Adding a legend
        #fig.legend(loc='upper left')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), bbox_transform=fig.transFigure)

        plt.title('Loss - Accuracy Plot')
        fig.tight_layout()
        plt.savefig(os.path.join(exp_dir, f'train_loss-acc-plot_iter-{global_iter}.png'))
        #plt.show()
        plt.clf() # Clear the plot
        plt.cla()
        logger.log(f'![Loss Accuracy Plot](train_loss-acc-plot_iter-{global_iter}.png)')
    elif mode == 'test':
        logger.log("**Plotting Testing Stats...**")
        iterations = np.arange(len(stats[list(stats.keys())[0]]))

        # Plotting the first series
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('% Accuracies / mIoU', color=color)
        ax1.plot(iterations, stats['test_accs'], marker='o', linestyle=":", color=color, label='% Test Acc')
        #ax1.plot(iterations, stats['test_mious'], marker='o', linestyle="-", color="tab:cyan", label='% Test mIoU')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=0, top=100)

        # Creating a secondary y-axis for the second series
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Losses', color=color)
        ax2.plot(iterations, stats['test_losses'], marker='o', color=color, label='Test Losses')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)

        # Creating a secondary y-axis for the second series
        ax3 = ax1.twinx()
        color = 'tab:cyan'
        ax3.set_ylabel('Testing mIoU', color=color)
        ax3.plot(iterations, stats['test_mious'], marker='o', linestyle="-", color="tab:cyan", label='% Test mIoU')
        #ax2.plot(epochs, per_epoch_stats['test_losses'], marker=',', linestyle=":", color="tab:pink", label='Test Losses')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim(bottom=0, top=100)
        # To prevent the two y-axes from overlapping
        ax3.spines['right'].set_position(('outward', 60))  # Offset the third axis
        ax3.spines['right'].set_color('green')

        # Adding a legend
        #fig.legend(loc='upper left')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), bbox_transform=fig.transFigure)

        plt.title('Loss - Accuracy Plot for Testing')
        fig.tight_layout()

        plt.savefig(os.path.join(exp_dir, f'test_loss-acc-plot_iter-{global_iter}.png'))
        #plt.show()
        plt.clf() # Clear the plot
        plt.cla()
        logger.log(f'![Loss Accuracy Plot](test_loss-acc-plot_iter-{global_iter}.png)')

def save_dict_to_json(data, path:str):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)
    return
    
def load_json_to_dict(path:str):
    with open('data.json', 'r') as json_file:
        data = json.load(path)
    return data

# Used to combine per_epoch confidence scores of train pts for data pruning
def standard_deviation_gpu(confidence_scores):
    """
    Calculate the standard deviation of confidence scores using GPU via PyTorch.

    Args:
    - confidence_scores (numpy array): Array of confidence scores with shape (epochs, num_points).

    Returns:
    - std_dev_scores (numpy array): Standard deviation of the scores for each data point.
    
    # Example usage
    confidence_scores = np.array([
        [0.9, 0.8, 0.7, 0.6],  # Scores for epoch 1
        [0.8, 0.7, 0.6, 0.5],  # Scores for epoch 2
        [0.85, 0.75, 0.65, 0.55]  # Scores for epoch 3
    ])

    std_dev_scores = standard_deviation_gpu(confidence_scores)
    print("Standard Deviation Scores:", std_dev_scores)`
    """
    # Convert the numpy array to a torch tensor and move it to the GPU
    scores_tensor = torch.tensor(confidence_scores, device='cuda')

    # Calculate the standard deviation along the first dimension (epochs)
    std_dev_tensor = torch.std(scores_tensor, dim=0)

    # Move the result back to the CPU and convert to a numpy array
    return std_dev_tensor.cpu().numpy()

def get_complement_indices(total_size, indices):
    """
    Generate an array of indices that are not in the given array of indices.

    Args:
    - total_size (int): The total number of elements in the original array.
    - indices (numpy array): An array of indices to exclude.

    Returns:
    - complement_indices (numpy array): An array of indices not in the given array of indices.
    """
    # Create a boolean mask with True for all indices
    mask = np.ones(total_size, dtype=bool)
    # Set the indices to exclude to False
    try:
        mask[indices] = False
    except Exception as e:
        print('Following error encountered in function : utils.get_complement_indices():\n', e)
        print(f'Total size = {total_size}')
        print(f'Type of array used as indices = {type(indices)}')
        print(f'Indices = {indices}')
        exit()
    # Return the indices where the mask is True
    complement_indices = np.nonzero(mask)[0]
    return complement_indices

def find_indices(main_array:np.ndarray, subset_array:np.ndarray) -> np.ndarray:
    """
    This function returns the indices at which elements of one array are present in another array.

    Arguments:
        main_array : reference array with respect to which the indices will be returned
        subset_array : the array whose elements will be searched for in the other array

    Returns:
        indices : indices at which elements of subset_array are present in the main_array
    """
    # Ensure both arrays are numpy arrays
    main_array = np.array(main_array)
    subset_array = np.array(subset_array)
    
    # Find the indices of elements in the main array that match the subset array
    indices = np.where(np.isin(main_array, subset_array))[0]
    
    return indices

def plot_cm(plot_save_path:str, class_names:list, predicted_labels:np.ndarray, ground_truth_labels:np.ndarray, super_title:str='Sample Super Title', title:str='Sample Title'):
    '''
    This function plots the confusion matrix.

    Arguments
        - plot_save_path : the path where plots will be saved
        - class_names : a list of class names eg ['class_1', 'class_2', 'class_3', ]
        - predicted_labels : numpy array of predicted labels eg [0, 1, 2, 2, 1, 0, 0, 0, 1, ] here 0 = 'class_1', 1 = 'class_2', ...
        - ground_truth_labels : numpy array of ground truths similar to predicted_labels
        - super_title : this is title of the graph shown at the top
        - title : this is a subtitle for the graph, shown below super_title

    Returns : None
    
    Example Code:
        ```python
            ground_truth_labels = np.random.randint(0,3,40)
            predicted_labels = np.random.randint(0,3,40)
            plot_save_path = 'test_cm.png'
            class_names = ['class_1', 'class_2', 'class_3']
            plot_cm(plot_save_path, class_names, predicted_labels, ground_truth_labels, 'Yo Bro!!', 'Yo Man!!')
        ```
    '''
    class_indices = [int(i) for i in range(len(class_names))]
    #print(class_indices)
    cm = confusion_matrix(predicted_labels, ground_truth_labels, labels=class_indices) # classes = list of class indices like [0, 1, 2, 3]
    df_cm = pd.DataFrame(np.int64(cm*100/np.max(cm)), index = class_names,
                        columns = class_names)
    plt.figure(figsize = (10,config.num_classes))
    seaborn.heatmap(df_cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.suptitle(super_title, fontdict={'color': 'black'})
    plt.title(title)
    plt.gca().invert_yaxis()
    #plt.show() 
    plt.savefig(plot_save_path)

@np.vectorize
def map_label(original_label:int):
    return config.label_mapping[original_label]

@np.vectorize
def inverse_map_label(label:int):
    return config.original_label_ids[label]

def colourize_image(remapped_array):
    colour_array = np.zeros(shape=(3,remapped_array.shape[0], remapped_array.shape[1]))
    for row in range(remapped_array.shape[0]):
        for col in range(remapped_array.shape[1]):
            colour_array[:,row,col] = config.colours[remapped_array[row, col]]

    return colour_array
