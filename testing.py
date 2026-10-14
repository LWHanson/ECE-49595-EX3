import torch
import training
from training import testCustom

# testCustom() uses training.model, so the trained weights must be loaded into it
training.model.load_state_dict(torch.load("classifier.pth", map_location=training.device))
training.model.to(training.device)

testCustom()
