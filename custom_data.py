from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from data_prep import BATCH_SIZE

IMG_SIZE = 32
IMG_HT = int(IMG_SIZE * 4 / 3)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_HT)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

imageset = datasets.ImageFolder("./custom_data", transform=transform)
set_loader = DataLoader(imageset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(imageset.classes)