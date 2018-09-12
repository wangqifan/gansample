
import torch.nn as nn

"""
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.3),
            nn.Linear(256,256),
            nn.LeakyReLU(0.3),
            nn.Linear(256,784),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.gen(x)
        x=x.view(-1,1,28,28)
        return x
"""
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.liner=nn.Sequential(
            nn.Linear(100,784),
            nn.ReLU()
        )
        self.conv=nn.Sequential(
              nn.Conv2d(1,12,3,1,padding=1),
              nn.ReLU(),
              nn.Conv2d(12,8,3,1,padding=1),
              nn.ReLU(),
              nn.Conv2d(8,1,3,1,padding=1)
        )
    def forward(self,x):
        x=self.liner(x)
        x=x.view(-1,1,28,28)
        x=self.conv(x)
        return x
