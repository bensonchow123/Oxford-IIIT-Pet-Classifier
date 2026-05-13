import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm
import torch.nn as nn

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

# Training:
# Resblock but focus on specific channels, residual connection fixes gradient vanquishing during backprobagation
# ResNet: https://arxiv.org/abs/1512.03385
# SENet: https://arxiv.org/abs/1709.01507
class SEResBlock(nn.Module):
    def __init__(self, channels, reduction=16): # in the SENet paper 16 is default and I tested others, this gives highest accuracy
        super().__init__()
        # The Resblock, this is just exta layers not the actual residual connection
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), # no relu here, want output to go to the residual conneciton first
        )
        # the SE block (squeeze and excitation branch)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # squeeze each channel to single num
            nn.Flatten(), # reshapes so it can go into linear layer
            nn.Linear(channels, channels // reduction), # actual excitation bottle neck
            nn.ReLU(), 
            nn.Linear(channels // reduction, channels), # expends back to per channel weights
            nn.Sigmoid() # squashed to 0 to 1
        )
        self.relu = nn.ReLU() # define the relu activation function

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

model_save_path = "/home/userfs/c/cqh514/Documents/Oxford-IIIT-Pet-Classifier/pet_classifier_weights.pth"
device = torch.device("cuda") 

test_tfms = v2.Compose([ # this needs to be same as the valuation transform
    v2.ToImage(),
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load the testing set
raw_test_data = datasets.OxfordIIITPet(root="/shared/storage/cs/studentscratch/cqh514", split='test', download=True)

# create the testing dataloader
test_dataset = TransformWrapper(raw_test_data, transform=test_tfms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

# define model and load the weights
model = OxfordPetClassifierCNN(num_classes=37).to(device)
model.load_state_dict(torch.load(model_save_path, weights_only=False))

# set to evaulation mode disable the batch norm
model.eval()

correct_test = 0
total_test = 0

print("Starting evaluation on the test dataset...")

# disable gradient calculations for testing, basicly same code as the validation phase in the training loop
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_acc = correct_test / total_test
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")