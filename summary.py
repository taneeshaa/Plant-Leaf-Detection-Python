from torchvision import models
from torchsummary import summary
import torch

modelLeaf = torch.load('best.pt')

print(modelLeaf)


