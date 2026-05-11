import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import CutMix, MixUp
from tqdm import tqdm

# Pretrianing:
# Training data transforms and augmentation
training_tfms_and_agmt = v2.Compose([
    v2.ToImage(), # this turns PIL images into torchvision tensor format

    v2.RandomRotation(degrees=15), # for realistic pet location variations 
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1)), # for better position accuracy
    v2.RandomHorizontalFlip(p=0.5), # it is still the same cat if flip like this but double data

    v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0), antialias=True), # 224 * 224 seems to be what everyone doing
    v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05), # this is for pet photographic varience

    v2.ToDtype(torch.float32, scale=True), # apply the image to float32 and scale from 0-255 to num between 0 and 1
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # apply the standardization for the z-scores things
])

# this let you do different transforms for train and validation sets
class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        super().__init__() # inherit the Dataset init method just in case even though I think there is nothing
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# load the trainval dataset for training
raw_data = datasets.OxfordIIITPet(root="/shared/storage/cs/studentscratch/cqh514", split='trainval', download=True)

# apply transforms and create a dataset
training_dataset = TransformWrapper(raw_data, transform=training_tfms_and_agmt)

# shuffle the data and use 10 cpu cores
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True, num_workers=10)

# save the model weights to my 5G of storage
model_save_path = "/home/userfs/c/cqh514/Documents/Oxford-IIIT-Pet-Classifier/pet_classifier_weights.pth"
device = torch.device("cuda") # cuda:1 is gpu 1, cuda:2 is gpu 2, etc

# Training:
# Resblock but focus on specific channels, fixes gradient vanquishing during backprobagation
class SEResBlock(nn.Module):
    def __init__(self, channels, reduction=16): # in the paper it says 16 is defaultly good and I tested others, this is best
        super().__init__()
        # The Resblock, this is just exta layers
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), # no relu here
        )
        # the SE block (squeeze and excitation branch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # squeeze each feature to single num
            nn.Flatten(), # reshapes so it can go into linear layer
            nn.Linear(channels, channels // reduction), # actual excitation bottle neck
            nn.ReLU(),
            nn.Linear(channels // reduction, channels), # expends back to per channel weights
            nn.Sigmoid() # squashed to 0 to 1
        )
        self.relu = nn.ReLU() # define the relu

    def forward(self, x):
        out = self.block(x)
        # reweight the images
        scale = self.se(out).view(x.size(0), -1, 1, 1) 
        # the actual residual connection
        return self.relu(x + out * scale) # normal resblock skip connection + reweighted each channel by importance

# main CNN inherits nn.Module
class OxfordPetClassifierCNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()
        
        # Convolutional Layers, adding more then 6 seems to decrease accuracy
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3), # resnet also uses initial kernel size of 7 and it seems to increase accuracy
            nn.BatchNorm2d(32), # normalize the inputs of each layer
            nn.ReLU(), # activation function
            nn.MaxPool2d(2, 2), # half the sptial size

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEResBlock(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEResBlock(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SEResBlock(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SEResBlock(512),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            SEResBlock(1024),
            nn.MaxPool2d(2, 2),
        )
        
        # takes feature map and turn into the 37 breed predictions
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # make it into a single average number
            nn.Flatten(), # remove all the empty dimensions
            nn.Linear(1024, num_classes) # map the 1024 features to 37 breed scores
        )

    def forward(self, x):
        x = self.features(x) # turn into feature map
        output = self.classifier(x) # 37 raw numbers for the possibilities
        return output

model = OxfordPetClassifierCNN().to(device)
print(model)

# use the cross entropy loss function with label smoothing, instead of like 1 or 0 for classification, have like 10 percent for other classes
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Adam optimizer for the loss function, weight decay prevents model relying on single feature
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)

EPOCHS = 30 # max amount of epoch i allowed to use
NUM_CLASSES = 37 # num of breed of cat and dogs
MIXUP_OR_CUTMIX_START_EPOCH = 20 # mixup and cutmix need to work on the batches, but starting from scratch lowers accuracy, 15 - 20 is best

# Tell the scheduler exactly how many total steps there are
total_steps = len(training_loader) * EPOCHS

# The one cycle Scheduler, basicly peak learning rate at epoch 9 and slow down by epoch 30, so it will be best for my 30 epoch limit
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3, # peak learning rate
    total_steps=total_steps, # the gradient steps for 30 epochs
    pct_start=0.3, # 30% of the epoches is warm up 
    div_factor=25.0, # div by this and you got the starting learning rate
    final_div_factor=1000.0 # div peak lr by this and get ending learning rate
)

# define the cutmix and mixup image augmentations
cutmix = CutMix(num_classes=NUM_CLASSES, alpha=0.1) # just enough to prevent overfitting, tested to be best
mixup = MixUp(num_classes=NUM_CLASSES, alpha=0.1) # just enough to prevent overfitting, tested to be best
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup]) # use it randomly on batches to prevent overfitting

# training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-" * 15)
    start_time = time.time() # for seeing how long a epoch takes and might just use another gpu
    
    # training starts here
    model.train() # this basicly activates the dropout but i removed it now it just turn on and off the batch norm
    # reset stuff after each epoch
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    use_augmentation = epoch >= MIXUP_OR_CUTMIX_START_EPOCH # determine to use the batch augmentations or not
    
    for inputs, labels in tqdm(training_loader, desc="Training", leave=False): # make it update instead of new line for the bar
        inputs, labels = inputs.to(device), labels.to(device) # move data to the vram 
        
        if use_augmentation:
            inputs, labels = cutmix_or_mixup(inputs, labels) # add augmentations to batches
        
        optimizer.zero_grad() # clear gradient from last batch
        outputs = model(inputs) # forward the data
        loss = criterion(outputs, labels) # run loss function
        loss.backward() # back probagation
        
        optimizer.step() # update the weights 
        
        scheduler.step() # update the learning rate for the one cycle lr
        
        running_loss += loss.item() * inputs.size(0) # loss per epoch
        _, predicted = torch.max(outputs, 1) # get the prediciton, discard the score
        total_train += labels.size(0) # get total images
        
        if use_augmentation:
            correct_train += (predicted == labels.argmax(dim=1)).sum().item() # add up the softlabels for correct train
        else:
            correct_train += (predicted == labels).sum().item() # count correct predictions
        
    train_acc = correct_train / total_train # the training accuracy
    current_lr = optimizer.param_groups[0]['lr'] # current learning rate
    
    print(f"LR: {current_lr:.5f} | Train Acc: {train_acc * 100:.2f}% | Time: {time.time() - start_time:.0f}s")
    
# save the model after all epochs finish
torch.save(model.state_dict(), model_save_path)

# signal that training is done
print(f"🌟 Saved final model at epoch {EPOCHS} with Train Acc: {train_acc * 100:.2f}%")
print(f"\nTraining done!")

