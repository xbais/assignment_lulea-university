from headers import *
from config import *
import config
import dataset
from losses import *
from networks import *
import utils
import networks

# Globals
per_epoch_stats = []
train_dataset, val_dataset, test_dataset = dataset.RellisDataset(), dataset.RellisDataset(), dataset.RellisDataset()

train_dataset.set_split('train')
val_dataset.set_split('val')
test_dataset.set_split('test')

#train_dataset.get_unique_labels()

'''
def plot_confusion_matrix(predictions, targets, num_classes, epoch):
    """
    Plot the confusion matrix using seaborn.
    
    :param predictions: numpy array of predicted class labels.
    :param targets: numpy array of true class labels.
    :param num_classes: Number of classes.
    :param epoch: Current epoch number.
    """
    cm = confusion_matrix(targets, predictions, labels=np.arange(num_classes))
    plt.figure(figsize=(10, num_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Epoch {epoch+1}')
    plt.show()
'''

def test_image(image_path:str, model, device, epoch:int=0):
    im = Image.open(image_path)
    im_array = np.array(im.resize(tuple(config.image_dims)))
    input = torch.unsqueeze(torch.tensor(im_array).permute(2, 0, 1).to(device).float(), dim=0)
    output = utils.logits2preds(model(input).cpu())

    # Save raw output image
    out_array = np.array(Image.fromarray(torch.squeeze(output).numpy().astype(np.uint8)).resize(im.size, resample=Image.BILINEAR)).astype(np.uint8)
    print(np.min(out_array), np.max(out_array))
    _out = utils.inverse_map_label(out_array)
    imageio.imwrite(os.path.join(config.BASE_DIR, f'raw_segmented_epoch={epoch}.png'), _out.astype(np.uint8))

    # Save coloured output image
    coloured = Image.fromarray(utils.colourize_image(out_array).transpose(1, 2, 0).astype(np.uint8)) #np.vectorize(utils.inverse_map_label)(np.expand_dims(out_array, axis=0))
    coloured.save(os.path.join(config.BASE_DIR, f'colour_segmented.png_epoch={epoch}.png'), 'PNG')

def train(loaders, model, optimizer, num_epochs, num_classes=config.num_classes, device='cuda', mode:str='train', plot_cm:bool=True):
    # Set the model to training mode
    
    torch.set_grad_enabled(True if mode=='train' else False)
    model = model.to(device)
    if mode == 'train':    
        model.train()
    elif mode == 'val':
        model.eval()

    # Loss function (CrossEntropyLoss for multi-class segmentation)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs if mode=='train' else 1):
        torch.set_grad_enabled(True if mode=='train' else False)
        running_loss = 0.0
        
        _total_step = math.ceil(len(train_dataset)/config.train_batch_size) if mode=='train' else math.ceil(len(val_dataset)/config.val_batch_size)
        
        tp, fp, fn, tn = np.zeros((len(config.label_names),)), np.zeros((len(config.label_names),)), np.zeros((len(config.label_names),)), np.zeros((len(config.label_names),))
        cm = np.zeros((len(config.label_names),len(config.label_names)))

        torch.cuda.empty_cache()

        # Loop through the dataloader (batches of images and labels)
        for i, (inputs, targets) in tqdm(enumerate(loaders[mode]), total=_total_step, leave=False):
            inputs = inputs.to(device).float()
            targets = targets.to(device).long()  # Ensure targets are LongTensor

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, targets)

            if mode=='train':
                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

            # Accumulate loss for display
            running_loss += loss.item()

            # Convert model outputs to predicted class labels
            _, preds = torch.max(outputs, 1)
            tp_, tn_, fp_, fn_, cm_ = utils.get_iou_stats(preds.cpu().numpy().flatten(), targets.cpu().numpy().flatten())
            tp += np.nan_to_num(tp_)
            fp += np.nan_to_num(fp_)
            tn += np.nan_to_num(tn_)
            fn += np.nan_to_num(fn_)
            cm += np.nan_to_num(cm_)

            # Print statistics every few batches
            if i % 10 == 9:    # Print every 10 batches
                #tqdm.write(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        class_ious = []
        for i in range(num_classes):
            class_iou = np.nan_to_num(tp[i]/(fp[i]+tp[i]+fn[i]))
            class_ious.append(class_iou)
        rich.print(', '.join([f'{config.label_names[_]} : {round(class_ious[_],2)}' for _ in range(num_classes)]))
            
        miou = np.mean(class_ious)*100
        print(f'{mode.upper()} MIOU = {miou}\n\n')
        if plot_cm:
            df_cm = pd.DataFrame(np.int64(cm*100/np.max(cm)), index = [i for i in label_names],
                                columns = [i for i in label_names])
            plt.figure(figsize = (10,7))
            seaborn.heatmap(df_cm, annot=True, fmt='d')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.title(f'Confusion Matrix (Percentages) | miou = {round(miou,2)}')
            plt.gca().invert_yaxis()
            #plt.show()
            plt.savefig(f'{mode}-cm_epoch-{epoch if mode=="train" else time.ctime()}.png')

        if mode == 'train':
            # Save trained model
            torch.save(model.state_dict(), f'trained-model_epoch-{epoch}.pt')

            # validate
            train(loaders=loaders, model=model, optimizer=optimizer, num_epochs=1, num_classes=config.num_classes, device='cuda', mode='val')
        
            # Test  
            test_image(image_path=config.test_image_path, model=model, device=device, epoch=epoch)

    print(f"{mode.upper()} completed.")

        
def get_loaders():  
    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=config.train_batch_size, 
                                            shuffle=False, # We shuffle points manually in dataset class
                                            drop_last=False,
                                            num_workers=config.dataloader_num_workers,
                                            generator=torch.Generator(device='cpu'),
                                            pin_memory=True, # Experimental
                                            timeout=config.dataloader_timeout,
                                            prefetch_factor=config.dataloader_prefetch_factor,
                                            ),
        'val' : torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=config.val_batch_size, 
                                            shuffle=False,
                                            drop_last=False, 
                                            num_workers=config.dataloader_num_workers,
                                            generator=torch.Generator(device='cpu'),
                                            pin_memory=True, # Experimental
                                            timeout=config.dataloader_timeout,
                                            prefetch_factor=config.dataloader_prefetch_factor,
                                            ),
        'test' : torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=config.test_batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=config.dataloader_num_workers,
                                            generator=torch.Generator(device='cpu'),
                                            pin_memory=True, # Experimental
                                            timeout=config.dataloader_timeout,
                                            prefetch_factor=config.dataloader_prefetch_factor,
                                            ),
    }
    return loaders