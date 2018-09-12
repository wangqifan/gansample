from torchvision import datasets
from torchvision import transforms 
import torch


batch_size=128


img_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])

mnist=datasets.MNIST(
    root='./data',train=True,transform=img_transform,download=True
    )

dataloader=torch.utils.data.DataLoader(
    dataset=mnist,batch_size=batch_size,shuffle=True
)

  
