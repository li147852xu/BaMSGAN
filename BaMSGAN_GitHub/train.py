import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from tqdm import tqdm
from model import Generator, Discriminator
from losses import Wasserstein, Hinge, Hinge1
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from utils import samimg,img_repo,rand_del

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(15)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128)
parser.add_argument('--blurratio', type=int, default=0.1)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--nz',
                    type=int,
                    default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch',
                    type=int,
                    default=300,
                    help='number of epochs to train for')
parser.add_argument('--lrd',
                    type=float,
                    default=2e-4,
                    help="Discriminator's learning rate, default=0.00005"
                    ) 
parser.add_argument('--lrg',
                    type=float,
                    default=2e-4,
                    help="Generator's learning rate, default=0.00005"
                    )  
parser.add_argument('--data_path',
                    default='./data',
                    help='folder to train data')  
parser.add_argument(
    '--outf',
    default='img/',
    help='folder to output images and model checkpoints') 
parser.add_argument('--blur_path',
                    default='./data_blur',
                    help='folder to store blur data')
parser.add_argument('--his_path',
                    default='./data_his',
                    help='folder to store history data')
opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([opt.imageSize, opt.imageSize]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)
dataset_blur = torchvision.datasets.ImageFolder(opt.blur_path,
                                                transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

dataloader_blur = torch.utils.data.DataLoader(
    dataset=dataset_blur,
    batch_size=round(opt.blurratio*opt.batchSize),
    shuffle=True,
    drop_last=True,
)


netG = Generator().to(device)
netG.apply(weights_init)
print('Generator:')
print(sum(p.numel() for p in netG.parameters()))

netD = Discriminator().to(device)
netD.apply(weights_init)
print('Discriminator:')
print(sum(p.numel() for p in netD.parameters()))

print(dataset, dataloader_blur)
start_epoch = 0  #设置初始epoch大小
#netG.load_state_dict(torch.load('img1/netG_0300.pth', map_location=device))  # 这两句用来读取预训练模型
#netD.load_state_dict(torch.load('img1/netD_0300.pth', map_location=device))  # 这两句用来读取预训练模型
criterionG = Hinge()
optimizerG = torch.optim.RMSprop(netG.parameters(),
                                 lr=opt.lrg,
                                 weight_decay=0.0001)  
optimizerD = torch.optim.RMSprop(netD.parameters(),
                                 lr=opt.lrd,
                                 weight_decay=0.0001)

lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD,
                                                           T_max=5,
                                                           eta_min=5E-5)
lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG,
                                                           T_max=5,
                                                           eta_min=5E-5)

criterionD = Hinge()
criterionD1 = Hinge1()
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
total_lossD = 0.0
total_lossG = 0.0
label = label.unsqueeze(1)


i=0
fixed_noise = torch.randn(opt.batchSize,100, 1, 1).to(device) 
for epoch in range(start_epoch + 1, opt.epoch + 1):
    with tqdm(total=len(dataloader),
              desc=f'Epoch {epoch}/{opt.epoch}',
              postfix=dict,
              mininterval=0.3) as pbar:
        for ((imgs, _), (imgs_blur, _)) in zip(dataloader, dataloader_blur):
            imgs = imgs.to(device)
            imgs_blur = imgs_blur.to(device)
            if i % 1 == 0:
                outputreal = netD(imgs)
                optimizerD.zero_grad()  
                noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
                noise = noise.to(device)
                fake = netG(noise)  
                outputfake = netD(fake.detach()) 
                outputfake_blur = netD(imgs_blur)
                if epoch>20:
                    for num in range(round(opt.batchSize*opt.blurratio)):
                        n = torch.randn((1, opt.nz, 1, 1)).to(device)
                        f = netG(n)
                        img_repo(f, epoch, num)
                if epoch>25:
                    imgs_his = samimg(round(opt.blurratio*opt.batchSize)).to(device)
                    outputfake_his = netD(imgs_his)
                    lossD = criterionD1(outputreal, outputfake, outputfake_blur,outputfake_his)
                else:
                    lossD = criterionD(outputreal, outputfake, outputfake_blur)
                total_lossD += lossD.item()
                lossD.backward()
                optimizerD.step()
                lrd_scheduler.step()
            if i % 5 == 0:  
                noise = torch.randn(opt.batchSize, opt.nz)
                noise = noise.to(device)
                fake = netG(noise)  
                optimizerG.zero_grad()
                output = netD(fake)
                lossG = criterionG(output)
                total_lossG += lossG.item()
                lossG.backward()
                optimizerG.step()

            pbar.set_postfix(
                **{
                    'total_lossD': total_lossD / ((i + 1)),
                    'lrd': get_lr(optimizerD),
                    'total_lossG': total_lossG / ((i + 1)),
                    'lrg': get_lr(optimizerG)
                })
            pbar.update(1)

    lrg_scheduler.step()
    fake = netG(fixed_noise)   
    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    log = open("./log.txt", 'a')
    print('[%d/%d] total_Loss_D: %.3f total_Loss_G %.3f' %
          (epoch, opt.epoch, total_lossD / (len(dataloader)), total_lossG /
           ((len(dataloader)))),
          file=log)
    total_lossG = 0.0
    total_lossD = 0.0
    log.close()
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/netG_%04d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%04d.pth' % (opt.outf, epoch))
