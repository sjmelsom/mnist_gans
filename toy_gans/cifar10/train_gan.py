import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.exposure import rescale_intensity


class D(nn.Module):
    def __init__(self, in_channels):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 64*2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, 64*4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(self.conv3.out_channels, 1, 4, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(64*2)
        self.bn2 = nn.BatchNorm2d(64*4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return torch.sigmoid(x).squeeze()

class G(nn.Module):
    def __init__(self, input_size, output_channels):
        super(G, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_size, 64*4, 4, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(self.conv1.out_channels, 64*2, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(self.conv2.out_channels, 64, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(self.conv3.out_channels, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64*4)
        self.bn2 = nn.BatchNorm2d(64*2)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.tanh(self.conv4(x))


class CIFAR10GAN:
    def __init__(self, data_root, debug, cuda_enabled, quiet, checkpoint):
        # Hyperparams
        self.batch_size = 128
        self.epochs = 200
        self.z_dim = 100
        self.lr = 2e-4

        self.cuda_enabled = cuda_enabled
        self.quiet = quiet
        self.checkpoint_dir = 'checkpoints'
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loaders
        self.train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.num_channels = 1 if self.train_dataset.data.ndim == 3 else self.train_dataset.data.shape[3]

        # Setup discriminator
        self.d = D(self.num_channels)
        if self.cuda_enabled:
            self.d.cuda()

        # Setup generator
        self.g = G(self.z_dim, self.num_channels)
        if self.cuda_enabled:
            self.g.cuda()

        # Load checkpoint
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)
            self.g.load_state_dict(state_dict['g'])
            self.d.load_state_dict(state_dict['d'])

        # Setup loss and optimizers
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)

        # Setup options
        if debug:
            self.epochs = 1

    def train_d(self, x):
        self.d.zero_grad()

        if self.cuda_enabled:
            x_real = x.cuda()

        # Create soft labels for real examples
        soft_labels = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1).squeeze()
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Pass real examples through discriminator
        d_output = self.d(x_real)

        # Compute discriminator loss for real examples
        real_loss = self.loss(d_output, soft_labels)
        
        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim, 1, 1).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Generate generated examples
        generated_images = self.g(z)
        if self.cuda_enabled:
            generated_images = generated_images.cuda()

        # Create labels for generated examples
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1).squeeze()
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Pass generated examples through discriminator
        generated_images = self.d(generated_images)
        generated_loss = self.loss(generated_images, soft_labels)

        # Add loss of real and generated examples to create the total loss
        d_loss = real_loss + generated_loss
        d_loss.backward()
        self.d_opt.step()

        return d_loss.data.item()

    def train_g(self, x):
        self.g.zero_grad()

        # Create labels
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1).squeeze()
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim, 1, 1).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Pass noise through generator to create generated images
        generated_images = self.g(z)

        # Pass generated images through discriminator
        d_output = self.d(generated_images)

        # Compute how good the generated images look by how well they were able
        # to fool the discriminator.
        g_loss = self.loss(d_output, soft_labels)

        # Step optimizer
        g_loss.backward()
        self.g_opt.step()

        return generated_images[:10], g_loss.data.item()

    def save_checkpoint(self, epoch):
        state_dict = {
            'g': self.g.state_dict(),
            'd': self.d.state_dict()
        }
        torch.save(state_dict, os.path.join(self.checkpoint_dir, f'{epoch}_cifar10gan.pt'))

    def train(self):
        writer = SummaryWriter()
        g_opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_opt, 0.99999)

        # For each epoch
        for epoch in range(self.epochs):
            tbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f'Epoch: {epoch+1}/{self.epochs}', 
                disable=self.quiet
            )
            # For each batch
            for batch_index, (x, _) in tbar:
                d_loss = self.train_d(x)

                # Get generated images and their corresponding labels
                generated_images, g_loss = self.train_g(x)

                # Step schedulers
                g_opt_scheduler.step()

            # Record layer metrics in tensorboard
            sd = self.g.state_dict()
            for l in sd.keys():
                writer.add_histogram(l, sd[l].cpu().detach().numpy(), epoch)

            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

            # Keep track of metrics
            writer.add_scalar('D Loss', d_loss, epoch)
            writer.add_scalar('G Loss', g_loss, epoch)

            # Keep track of learning rate
            g_opt_lr = self.g_opt.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('Generator Learning Rate', g_opt_lr, epoch)

            # Save some example images
            image_grid = self.setup_images_for_tboard(generated_images)
            writer.add_image('Image', image_grid, epoch)
            
        # Close writer
        writer.close()

    def setup_images_for_tboard(self, generated_images):
        imgs = generated_images.view(-1, 3, 32, 32)
        grid = make_grid(imgs, nrow=5)
        return grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savegen', dest='save_model', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataroot', dest='data_root', default='../data')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--warmstart', dest='checkpoint')
    args = parser.parse_args()

    # Check for cuda
    cuda_enabled = torch.cuda.is_available()

    # Instantiate model
    gan = CIFAR10GAN(args.data_root, args.debug, cuda_enabled, args.quiet, args.checkpoint)

    # Train model
    gan.train()

    # Save model
    if args.save_model:
        torch.save(gan.g.state_dict(), 'cifar10_gen.pt')

