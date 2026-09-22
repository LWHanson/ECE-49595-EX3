# This script trains a convolutional neural network (CNN) for the image classification model

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
from data_prep import testing_loader
from data_prep import training_loader
from data_prep import image_loader
from data_prep import classes
from data_prep import BATCH_SIZE
from data_prep import NUM_LABELS

KERNEL_SIZE = 5
CHANNELS = 12
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PRINT_COUNT = 1000
TRAIN_SIZE = 5

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=CHANNELS, kernel_size=KERNEL_SIZE, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(CHANNELS)
        self.conv2 = nn.Conv2d(in_channels=CHANNELS, out_channels=CHANNELS, kernel_size=KERNEL_SIZE, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(CHANNELS)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=CHANNELS, out_channels=2*CHANNELS, kernel_size=KERNEL_SIZE, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2*CHANNELS)
        self.conv5 = nn.Conv2d(in_channels=2*CHANNELS, out_channels=2*CHANNELS, kernel_size=KERNEL_SIZE, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(2*CHANNELS)
        self.fc1 = nn.Linear(2*CHANNELS*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 2*CHANNELS*10*10)
        output = self.fc1(output)
        return(output)

model = Network()

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

def saveModel():
    path = "./classifier.pth"
    torch.save(model.state_dict(), path)

def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()

    accuracy = (100 * accuracy / total)
    return(accuracy)

def train(num_epochs):
    best_accuracy = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(training_loader, 0):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i % PRINT_COUNT) == (PRINT_COUNT - 1):
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / PRINT_COUNT))
                running_loss = 0.0

        accuracy = testAccuracy()
        print("For epoch " + str(epoch + 1) + ", the test accuracy over the test set is " + str(round(accuracy, 2)) + "%")

        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

def imageshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def testBatch():
    images, labels = next(iter(testing_loader))

    print("Actual labels: ", " ".join("%5s" % classes[labels[j]] for j in range(BATCH_SIZE)))
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(BATCH_SIZE)))
    imageshow(torchvision.utils.make_grid(images))

def testClasses():
    class_correct = list(0. for i in range(NUM_LABELS))
    class_total = list(0. for i in range(NUM_LABELS))

    with torch.no_grad():
        for data in testing_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(BATCH_SIZE):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(NUM_LABELS):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))

def testCustom():
    images, labels = next(iter(image_loader))

    print("Actual labels: ", " ".join("%5s" % classes[labels[j]] for j in range(BATCH_SIZE)))
    outputs = model(images)
    
    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(BATCH_SIZE)))
    imageshow(torchvision.utils.make_grid(images))
    # custom_correct = 0
    # custom_total = 0

    # with torch.no_grad():
    #     for images, labels in image_loader:
    #         predictions = model(images).argmax(dim=1)
    #         custom_correct += (predictions == labels).sum().item()
    #         custom_total += labels.size(0)
    #         print("Predicted Label: " + predictions)
    #         print("Actual Label: " + labels)
    # imageshow(torchvision.utils.make_grid(images))

if __name__ == "__main__":
    print("Starting Training")
    train(TRAIN_SIZE)
    print("Training Finished")

    testClasses()
    testAccuracy()

    model = Network()
    path = "classifier.pth"
    model.load_state_dict(torch.load(path))

    testBatch()