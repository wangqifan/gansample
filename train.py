import torch
import torch.nn as nn
import  torch.optim  as optim
from torch.autograd import Variable
from torchvision.utils import save_image


from LoadData import dataloader
from  discriminator import discriminator
from generator import generator
import os


if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    out=(x+1)*0.5
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

num_epoch=100
z_dimension = 100

Disc=discriminator()
Gen=generator()
Disc=Disc.cuda()
Gen=Gen.cuda()


criterion=nn.BCELoss()

d_optimizer=optim.Adam(Disc.parameters(),lr=0.001)
g_optimizer=optim.Adam( Gen.parameters(),lr=0.001)


for epoch in range(num_epoch):
    for i, (img, _) in enumerate(dataloader):
        num_img=img.size(0)
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()
        #优化disc--------------------
        for j in range(2):

            d_optimizer.zero_grad()
            noise=Variable(torch.randn(num_img,z_dimension)).cuda()
        
        
            fake_image=Gen(noise)
            fake_image.view(-1,1,28,28)

            real_out=Disc(real_img)
            fake_out=Disc(fake_image)

            real_out=real_out.view(-1)
            fake_out=fake_out.view(-1)
        
            r_loss=criterion(real_out,real_label)
            f_loss=criterion(fake_out,fake_label)

            d_loss=r_loss+f_loss
            d_loss.backward()
            d_optimizer.step()
        for j in range(2):

            g_optimizer.zero_grad()
            noise=Variable(torch.randn(num_img,z_dimension)).cuda()
            fake_image=Gen(noise)
            fake_image.view(-1,1,28,28)
            real_out=Disc(real_img)
            fake_out=Disc(fake_image)
            
            fake_images =to_img(fake_image.cpu().data)

            real_out=real_out.view(-1)
            fake_out=fake_out.view(-1)
        

            r_loss=criterion(real_out,real_label)
            f_loss=criterion(fake_out,real_label)
            
            g_loss=f_loss+r_loss
            g_loss.backward()
            g_optimizer.step()
    
        print("{:.6f},{:.6f}".format(g_loss.data[0],d_loss.data[0]))
    
    if epoch == 0:
        real_images =to_img(real_img.cpu().data)
        save_image(real_images, './img/real_images.png')
    fake_images =to_img(fake_image.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

torch.save(Gen,"G.pkl")
torch.save(Disc,"D.pkl")



