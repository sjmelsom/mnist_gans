import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from skimage.io import imsave
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from skimage.exposure import rescale_intensity


class D(nn.Module):
    def __init__(self, input_size):
        super(D, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x), 0.2)
        x = F.relu(self.fc3(x), 0.2)
        return torch.sigmoid(self.fc4(x))

class G(nn.Module):
    def __init__(self, input_size, n_class):
        super(G, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x), 0.2)
        x = F.relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class MNISTGAN:
    def __init__(self, experiment):
        self.batch_size = 128
        self.lr = 2e-4
        self.epochs = 100
        self.z_dim = 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        self.train_dataset = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.mnist_dim = self.train_dataset.train_data.size(1) * self.train_dataset.train_data.size(2)
        self.d = D(self.mnist_dim)
        self.d.cuda()
        self.g = G(self.z_dim, self.mnist_dim)
        self.g.cuda()
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)
        self.experiment = experiment
        exp_dir = os.path.join('experiments', experiment)
        self.images_dir = os.path.join(exp_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def train_d(self, x):
        self.d.zero_grad()

        x_real = x.view(-1, self.mnist_dim)
        y_real = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1)
        x_real = x_real.cuda()
        y_real = y_real.cuda()

        d_output = self.d(x_real)
        real_loss = self.loss(d_output, y_real)
        
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        z = z.cuda()
        x_fake = self.g(z)
        y_fake = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
        x_fake = x_fake.cuda()
        y_fake = y_fake.cuda()

        g_output = self.d(x_fake)
        fake_loss = self.loss(g_output, y_fake)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_opt.step()

        return d_loss.data.item()

    def train_g(self, x):
        self.g.zero_grad()

        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        # z = torch.randn(self.batch_size, self.z_dim)
        z = z.cuda()
        y = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1)
        y = y.cuda()

        g_output = self.g(z)
        d_output = self.d(g_output)
        g_loss = self.loss(d_output, y)

        g_loss.backward()
        self.g_opt.step()

        return g_output[:10], g_loss.data.item()


    def train(self):
        t1 = tqdm(range(self.epochs), position=0)
        for epoch in t1:
            for batch_index, (x, _) in tqdm(enumerate(self.train_loader), position=1, total=len(self.train_loader)):
                d_loss = self.train_d(x)
                g_output, g_loss = self.train_g(x)
            self.log_images(g_output, epoch)
            t1.set_description(f'D: {d_loss:.2f} | G: {g_loss:.2f}')


    def log_images(self, g_output, epoch):
        im_name = os.path.join(self.images_dir, f'epoch{epoch}.jpg')
        imgs = g_output.view(-1, 28, 28)
        imgs = [img.cpu().detach().numpy() for img in imgs]
        imgs = [rescale_intensity(img, out_range=(0, 255)).astype(np.uint8) for img in imgs]
        imgs = np.hstack(imgs)
        imsave(im_name, imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    args = parser.parse_args()
    gan = MNISTGAN(args.experiment)
    gan.train()

