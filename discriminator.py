import torch.nn as  nn


class discriminator(nn.Module):
    """
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
             nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(x.size(0),-1)
        x=self.dis(x)
        return x
    """
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Conv2d(1,12,3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(12,20,3),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.Conv2d(20,10,3),
            nn.MaxPool2d(2)
        )
        self.full=nn.Sequential(
            nn.Linear(10*5*5,120),
            nn.LeakyReLU(0.2),
            nn.Linear(120,32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x.view(-1,1,28,28)
        x=self.dis(x)
        x=x.view(-1,10*5*5)
        x=self.full(x)
        return x
