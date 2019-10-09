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
    def __init__(self, experiment, data_root):
        # Hyperparams
        self.batch_size = 128
        self.epochs = 100
        self.z_dim = 100
        self.lr = 2e-4

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loaders
        # self.train_dataset = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
        self.train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.mnist_dim = self.train_dataset.train_data.size(1) * self.train_dataset.train_data.size(2)

        # Setup discriminator
        self.d = D(self.mnist_dim)
        self.d.cuda()

        # Setup generator
        self.g = G(self.z_dim, self.mnist_dim)
        self.g.cuda()

        # Setup loss and optimizers
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)

        # Setup logging
        self.experiment = experiment
        '''
        exp_dir = os.path.join('experiments', 'gan', experiment)
        self.images_dir = os.path.join(exp_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        '''

    def train_d(self, x):
        self.d.zero_grad()

        x_real = x.view(-1, self.mnist_dim)
        x_real = x_real.cuda()
        soft_labels = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1)
        soft_labels = soft_labels.cuda()

        # Pass real examples through discriminator
        d_output = self.d(x_real)

        # Compute discriminator loss for real examples
        real_loss = self.loss(d_output, soft_labels)

        # Compute discriminator accuracy
        d_acc = (d_output.round() == soft_labels.round()).sum().float() / d_output.float().shape[0]
        
        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        z = z.cuda()

        # Generate generated examples
        generated_images = self.g(z)
        generated_images = generated_images.cuda()

        # Create labels for generated examples
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
        soft_labels = soft_labels.cuda()

        # Pass generated examples through discriminator
        generated_images = self.d(generated_images)
        generated_loss = self.loss(generated_images, soft_labels)

        # Add loss of real and generated examples to create the total loss
        d_loss = real_loss + generated_loss
        d_loss.backward()
        self.d_opt.step()

        return d_loss.data.item(), d_acc

    def train_g(self, x):
        self.g.zero_grad()

        # Create labels
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1)
        soft_labels = soft_labels.cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
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

    def train(self):
        t1 = tqdm(range(self.epochs), position=0)
        writer = SummaryWriter()

        # For each epoch
        for epoch in t1:
            d_accuracy = 0

            # For each batch
            for batch_index, (x, _) in enumerate(self.train_loader):
                d_loss, d_acc = self.train_d(x)

                # Accumulate d_acc
                d_accuracy += d_acc

                # Get generated images and their corresponding labels
                generated_images, g_loss = self.train_g(x)

            # Keep track of loss metrics
            writer.add_scalar('D Loss', d_loss, epoch)
            writer.add_scalar('G Loss', g_loss, epoch)

            # Keep track of discriminator accuracy
            writer.add_scalar('D Accuracy', d_accuracy/self.batch_size, epoch)

            # Save some example images
            image_grid = self.setup_images_for_tboard(generated_images)
            writer.add_image('Image', image_grid, epoch)
            
            # Update progress bar
            t1.set_description(f'D Loss: {d_loss:.2f} | G Loss: {g_loss:.2f}')

        # Close writer
        writer.close()

    def setup_images_for_tboard(self, generated_images):
        imgs = generated_images.view(-1, 1, 28, 28)
        grid = make_grid(imgs, nrow=5)
        return grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    parser.add_argument('--dataroot', dest='data_root', default='../data')
    args = parser.parse_args()
    gan = MNISTGAN(args.experiment, args.data_root)
    gan.train()

