import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from skimage.io import imsave
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.exposure import rescale_intensity


class D(nn.Module):
    def __init__(self, input_size, num_classes):
        super(D, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, num_classes)
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, x, labels):
        x = x.view(x.size(0), 28*28)
        c = self.label_emb(labels)
        c = torch.squeeze(c)
        x = torch.cat([x, c], 1)
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
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, x, labels):
        x = x.view(x.size(0), 100)
        c = self.label_emb(labels)
        c = torch.squeeze(c)
        x = torch.cat([x, c], 1)
        x = F.relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x), 0.2)
        x = F.relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        return x.view(x.size(0), 28, 28)


class MNISTCGAN:
    def __init__(self, data_root, debug, cuda_enabled, quiet):
        # Hyperparams
        self.batch_size = 128
        self.epochs = 150
        self.z_dim = 100
        self.lr = 1e-4

        self.cuda_enabled = cuda_enabled
        self.quiet = quiet

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loader
        self.train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

        num_digits = 10
        self.mnist_dim = self.train_dataset.data.size(1) * self.train_dataset.data.size(2)

        # Setup discriminator
        self.d = D(self.mnist_dim+num_digits, 1)
        if self.cuda_enabled:
            self.d.cuda()

        # Setup generator
        self.g = G(self.z_dim+num_digits, self.mnist_dim)
        if self.cuda_enabled:
            self.g.cuda()

        # Setup loss and optimizers
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)

        # Setup options
        if debug:
            self.epochs = 1

    def train_d(self, x, labels):
        self.d.zero_grad()

        labels = labels.long()

        if self.cuda_enabled:
            x = x.cuda()
            labels = labels.cuda()

        # Pass real examples through discriminator
        d_output = self.d(x, labels)
        soft_labels = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1)
        soft_labels = soft_labels.cuda()

        # Compute discriminator loss for real examples
        real_loss = self.loss(d_output, soft_labels)
        
        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Create labels
        labels = torch.LongTensor(self.batch_size, 1).random_(0, 10)
        if self.cuda_enabled:
            labels = labels.cuda()

        # Generate images
        generated_images = self.g(z, labels)

        # Pass generated examples through discriminator
        d_output = self.d(generated_images, labels)

        # Create labels for generated examples
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Compute loss for generated examples
        generated_loss = self.loss(d_output, soft_labels)

        # Add loss of real and generated examples to create the total loss
        d_loss = real_loss + generated_loss
        d_loss.backward()
        self.d_opt.step()

        return d_loss.data.item()

    def train_g(self, x):
        self.g.zero_grad()

        # Create random labels
        labels = torch.LongTensor(self.batch_size, 1).random_(0, 10)
        if self.cuda_enabled:
            labels = labels.cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(labels.shape[0], self.z_dim).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Pass noise and labels through generator to create generated images
        generated_images = self.g(z, labels)

        # Pass generated images and labels through discriminator
        d_output = self.d(generated_images, labels)

        # Create soft labels (near 0 or 1 instead of exactly 0 or 1)
        soft_labels = torch.FloatTensor(labels.shape[0], 1).uniform_(0, 0.1)
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Compute how good the generated images look by how well they were able
        # to fool the discriminator.
        g_loss = self.loss(d_output, soft_labels)
        if self.cuda_enabled:
            g_loss.cuda()

        # Step optimizer
        g_loss.backward()
        self.g_opt.step()

        return g_loss.data.item()

    def train(self):
        writer = SummaryWriter()

        for epoch in range(self.epochs):
            tbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f'Epoch: {epoch+1}/{self.epochs}', 
                disable=self.quiet
            )
            for batch_index, (x, y) in tbar:
                d_loss = self.train_d(x, y)

                # Get generated images and their corresponding labels
                g_loss = self.train_g(x)

            # Record layer metrics in tensorboard
            sd = self.g.state_dict()
            for l in sd.keys():
                writer.add_histogram(l, sd[l].cpu().detach().numpy(), epoch)

            # Keep track of metrics
            writer.add_scalar('D Loss', d_loss, epoch)
            writer.add_scalar('G Loss', g_loss, epoch)

            # Keep track of learning rate
            g_opt_lr = self.g_opt.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('Generator Learning Rate', g_opt_lr, epoch)

            # Save some example images
            image_grid = self.generate_examples()
            writer.add_image('Image', image_grid, epoch)

        # Close writer
        writer.close()

    def generate_examples(self):
        # Create latent noise vectors
        z = torch.FloatTensor(50, self.z_dim).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Create labels
        labels = torch.LongTensor(np.repeat(np.arange(0, 10), 5).reshape(10, 5).T.flatten())
        if self.cuda_enabled:
            labels = labels.cuda()
        images = self.g(z, labels)

        images = images.view(-1, 1, 28, 28)
        return make_grid(images, nrow=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--savegen', dest='save_model', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataroot', dest='data_root', default='../data')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    # Check for cuda
    cuda_enabled = torch.cuda.is_available()

    # Instantiate model
    gan = MNISTCGAN(args.data_root, args.debug, cuda_enabled, args.quiet)

    # Train model
    gan.train()

    # Save model
    if args.save_model:
        torch.save(gan.g.state_dict(), 'mnist_cgen.pt')
