# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import argparse
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
from utils import weights_init, compute_acc
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt

# %% [markdown]
#  # 输入参数

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--signalFeatures', type=int, default=6, help='the features of signal')
parser.add_argument('--signalPoints', type=int, default=6, help='the points of signal')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
parser.add_argument('--window', type=int, default=4, help='The size of slide window')
parser.add_argument('--slide_stride', type=int, default=1, help='The stride of slide window')

opt = parser.parse_args(['--cuda',
                         '--batchSize','500',
                         '--niter','500',
                         '--workers','0',
                         '--gpu_id','0',
                         '--nz','100',
                         '--num_classes','6',
                         '--signalFeatures','5',
                         '--signalPoints','4',
                         '--window','4',
                         '--slide_stride','1',
                         '--lr','0.0001'])
print(opt)

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

device = torch.device("cuda:"+str(opt.gpu_id) if torch.cuda.is_available() else "cpu")


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3


# %%
# 导入数据
data_1 = torch.from_numpy(scio.loadmat("mode_1_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)
data_2 = torch.from_numpy(scio.loadmat("mode_2_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)
data_3 = torch.from_numpy(scio.loadmat("mode_3_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)
data_4 = torch.from_numpy(scio.loadmat("mode_4_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)
data_5 = torch.from_numpy(scio.loadmat("mode_5_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)
data_6 = torch.from_numpy(scio.loadmat("mode_6_data.mat")['X_train']).type(torch.float32).unsqueeze(2).to(device)


# %%
# 滑窗

def slide_window_data(data, window, stride):
    # data: torch.ndarray samples*features*1
    new_data = data[:-window,:,:]
    for i in range(1,window):
        new_data = torch.cat((new_data,data[i:-window+i,:,:]),2)
    

    output = new_data[[x*stride for x in range((new_data.shape[0]-1)//stride+1)], :, :]

    return output


# %%
real_data_1 = slide_window_data(data_1, opt.window, opt.slide_stride)
real_data_2 = slide_window_data(data_2, opt.window, opt.slide_stride)
real_data_3 = slide_window_data(data_3, opt.window, opt.slide_stride)
real_data_4 = slide_window_data(data_4, opt.window, opt.slide_stride)
real_data_5 = slide_window_data(data_5, opt.window, opt.slide_stride)
real_data_6 = slide_window_data(data_6, opt.window, opt.slide_stride)


# %%
# 数据加载DataLoader
x_data = torch.cat((real_data_1,real_data_2,real_data_3,real_data_4,real_data_5,real_data_6),0)
label_1 = torch.zeros(real_data_1.shape[0])
label_2 = torch.ones(real_data_2.shape[0])
label_3 = torch.ones(real_data_3.shape[0])*2
label_4 = torch.ones(real_data_4.shape[0])*3
label_5 = torch.ones(real_data_5.shape[0])*4
label_6 = torch.ones(real_data_6.shape[0])*5
y_data = torch.cat((label_1,label_2,label_3,label_4,label_5,label_6))


# %%
# 临时 去除正常数据
# x_data = torch.cat((real_data_2,real_data_3,real_data_4,real_data_5,real_data_6),0)
# # label_1 = torch.zeros(real_data_1.shape[0])
# label_2 = torch.zeros(real_data_2.shape[0])
# label_3 = torch.ones(real_data_3.shape[0])*1
# label_4 = torch.ones(real_data_4.shape[0])*2
# label_5 = torch.ones(real_data_5.shape[0])*3
# label_6 = torch.ones(real_data_6.shape[0])*4
# y_data = torch.cat((label_2,label_3,label_4,label_5,label_6))


# %%
x_data = x_data[:,0:opt.signalFeatures,:]


# %%
deal_dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset=deal_dataset,
                        batch_size=opt.batchSize,
                        shuffle=True,
                        num_workers=opt.workers)


# %%
# 定义生成器
class _netG(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ngpu = int(opt.ngpu)
        self.nz = int(opt.nz)
        self.points = int(opt.signalPoints)
        self.batch_size = int(opt.batchSize)
        self.features = int(opt.signalFeatures)

        self.fc1 = nn.Linear(100,36)
        # output batchsize*36*1

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose1d(36,18,2,1,0, bias=False),
            nn.BatchNorm1d(18),
            nn.ReLU(True)
        )# output batchsize*channel(dim)*points batchsize*18*2

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose1d(18,opt.signalFeatures,2,1,0, bias=False),
            nn.BatchNorm1d(opt.signalFeatures),
            nn.ReLU(True)
        )# output batchsize*channel(dim)*points batchsize*6*4

        self.tconv4 = nn.Sequential(
            nn.ConvTranspose1d(9,opt.signalFeatures,self.points-2,1,0, bias=False),
            nn.BatchNorm1d(opt.signalFeatures),
            nn.ReLU(True)
        )# output batchsize*channel(dim)*points batchsize*9*3

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 36, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            output = tconv4
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 36, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            output = tconv4
        return output


# %%
# 定义鉴别器
class _netD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.ngpu = int(opt.ngpu)
        self.num_classes = opt.num_classes
        self.points = int(opt.signalPoints)
        self.features = int(opt.signalFeatures)

        self.conv1 = nn.Sequential(
            nn.Conv1d(opt.signalFeatures,12,2,1,0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )# output [1, 12, 3]
        self.conv2 = nn.Sequential(
            nn.Conv1d(12,24,2,1,0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )# output [1, 24, 1]
        self.conv3 = nn.Sequential(
            nn.Conv1d(24,48,2,1,0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )# output [1, 48, 1]

        self.fc_dis = nn.Linear(48*(self.points-3), 1)
        self.fc_aux = nn.Linear(48*(self.points-3), self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batchsize = input.shape[0]
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            flat3 = conv3.view(-1, 48*1)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat3, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat3, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            flat3 = conv3.view(batchsize, -1)
            fc_dis = self.fc_dis(flat3)
            fc_aux = self.fc_aux(flat3)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


# %%
# 初始化生成器和鉴别器
netG = _netG(opt).to(device)
netG.apply(weights_init)
netD = _netD(opt).to(device)
netD.apply(weights_init)


# %%
# 定义鉴别和辅助分类损失函数
dis_criterion = nn.BCELoss().to(device)
aux_criterion = nn.NLLLoss().to(device)

# %% [markdown]
#  # 定义优化器

# %%
# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# %% [markdown]
#  # 训练
# %% [markdown]
#  # 初始化变量

# %%
# tensor placeholders
noise = torch.FloatTensor(opt.batchSize, nz, 1).to(device)
dis_label = torch.FloatTensor(opt.batchSize).to(device)
aux_label = torch.LongTensor(opt.batchSize).to(device)
real_label = 1
fake_label = 0


# %%
avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0


# %%
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        if epoch==400:
            pass

        #############################
        ## (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #############################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        input_samples = real_cpu
        batch_size = real_cpu.shape[0]
        dis_label.resize_(batch_size).fill_(real_label)
        aux_label.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input_samples)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        D_x = dis_output.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        label = torch.randint_like(aux_label, 0, num_classes)
        noise = torch.randn(batch_size, nz,1,1).to(device)
        class_onehot = torch.zeros((batch_size, num_classes)).to(device)
        class_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :num_classes, 0, 0] = class_onehot[np.arange(batch_size)]
        aux_label.copy_(label)

        fake = netG(noise)
        dis_label.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        D_G_z1 = dis_output.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        #############################
        ## (2) Update G network: maximize log(D(G(z)))
        #############################
        netG.zero_grad()
        dis_label.fill_(real_label)
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + aux_errG
        errG.backward()
        D_G_z2 = dis_output.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = 1
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.item()
        all_loss_D += errD.item()
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)
        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                % (epoch, opt.niter, i, len(dataloader),
                errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))


# %%
label = torch.randint_like(aux_label, 0, num_classes)
label[:] = 1-1
noise = torch.randn(batch_size, nz,1,1).to(device)
class_onehot = torch.zeros((batch_size, num_classes)).to(device)
class_onehot[np.arange(batch_size), label] = 1
noise[np.arange(batch_size), :num_classes, 0, 0] = class_onehot[np.arange(batch_size)]
aux_label.copy_(label)

fake = netG(noise)
plot_fake_sample = fake.detach().cpu()
plt.figure()
plt.subplot(2,1,1)
plt.plot(plot_fake_sample[:,:,0])
plt.subplot(2,1,2)
plt.plot(data_5[:,:,0].detach().cpu())
plt.savefig('fig1.png')


# %%
torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))


# %%



