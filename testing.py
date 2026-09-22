import torch
import torchvision
from training import Network
from training import testCustom

net = Network()

net.load_state_dict(torch.load("classifier.pth"))

testCustom()