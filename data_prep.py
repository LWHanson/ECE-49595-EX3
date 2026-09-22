# This script obtains training and testing data for the image classification model

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

BATCH_SIZE = 10
NUM_LABELS = 10

training_set = CIFAR10(root="./data", train=True, transform=transformations, download=True)

training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print("Number of images in training set: " + str(len(training_loader) * BATCH_SIZE))

testing_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)    # shuffle=True selects a random set of images to test with
print("Number of images in testing set: " + str(len(testing_loader) * BATCH_SIZE))

print("Number of batches per epoch: " + str(len(training_loader)))
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")