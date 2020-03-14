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


class C(nn.Module):
    def __init__(self, input_size):
        super(C, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x), 0.2)
        x = F.relu(self.fc3(x), 0.2)
        return self.fc4(x)

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


class MNISTWGAN:
    def __init__(self, data_root, debug, cuda_enabled, quiet):
        # Hyperparams
        self.batch_size = 64
        self.epochs = 150
        self.z_dim = 100
        self.g_lr = 2e-4
        self.c_lr = 5e-6
        self.grad_clip = 0.01
        self.n_critic = 5

        self.cuda_enabled = cuda_enabled
        self.quiet = quiet

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loaders
        self.train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.mnist_dim = self.train_dataset.data.size(1) * self.train_dataset.data.size(2)

        # Setup critic
        self.c = C(self.mnist_dim)
        if self.cuda_enabled:
            self.c.cuda()

        # Setup generator
        self.g = G(self.z_dim, self.mnist_dim)
        if self.cuda_enabled:
            self.g.cuda()

        # Setup loss and optimizers
        self.g_opt = optim.RMSprop(self.g.parameters(), lr=self.g_lr)
        self.c_opt = optim.RMSprop(self.c.parameters(), lr=self.c_lr)

        # Setup options
        if debug:
            self.epochs = 1

    def train_c(self, x):
        self.c.zero_grad()

        # Add dimension to real examples
        x_real = x.view(-1, self.mnist_dim)
        if self.cuda_enabled:
            x_real = x_real.cuda()

        # Create labels for real and fake examples 
        real_label = torch.FloatTensor([1])
        fake_label = torch.FloatTensor([-1])
        if self.cuda_enabled:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        # Pass real examples through critic 
        c_real_scores = self.c(x_real)

        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Generate generated examples
        generated_images = self.g(z)
        if self.cuda_enabled:
            generated_images = generated_images.cuda()

        # Pass generated examples through critic
        c_fake_scores = self.c(generated_images)

        # Add loss of real and generated examples to create the total loss
        real_mean = c_real_scores.mean(axis=0)
        fake_mean = c_fake_scores.mean(axis=0)
        c_loss = torch.mean(torch.dot(real_label*real_mean, fake_label*fake_mean))
        c_loss.backward()
        self.c_opt.step()

        return c_loss.data.item()

    def train_g(self, x):
        self.g.zero_grad()

        # Create labels
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0, 0.1)
        if self.cuda_enabled:
            soft_labels = soft_labels.cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        if self.cuda_enabled:
            z = z.cuda()

        # Pass noise through generator to create generated images
        generated_images = self.g(z)

        # Pass generated images through critic
        c_score = self.c(generated_images)

        # Compute how good the generated images look by how well they were able
        # to fool the critic.
        g_loss = -c_score.mean()

        # Step optimizer
        g_loss.backward()
        self.g_opt.step()

        return generated_images[:10], g_loss.data.item()

    def train(self):
        writer = SummaryWriter()
        clipper = WeightClipper(self.grad_clip)

        # For each epoch
        for epoch in range(self.epochs):
            tbar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader), 
                desc=f'Epoch: {epoch+1}/{self.epochs}', 
                disable=self.quiet
            )
            # Iterate the critic and generator n_critic:1 (usually 5:1)
            for batch_index, (x, _) in tbar:
                if batch_index % self.n_critic != 0:
                    c_loss = self.train_c(x)
                    self.c.apply(clipper)

                if batch_index % self.n_critic == 0:
                    generated_images, g_loss = self.train_g(x)

            # Record layer metrics in tensorboard
            sd = self.g.state_dict()
            for l in sd.keys():
                writer.add_histogram(l, sd[l].cpu().detach().numpy(), epoch)

            # Keep track of metrics
            writer.add_scalar('C Loss', c_loss, epoch)
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
        imgs = generated_images.view(-1, 1, 28, 28)
        grid = make_grid(imgs, nrow=5)
        return grid


class WeightClipper(object):
    def __init__(self, clip_constant):
        self.cc = clip_constant

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.cc, self.cc)
            module.weight.data = w

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--savegen', dest='save_model', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataroot', dest='data_root', default='../data')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    # Check for cuda
    cuda_enabled = torch.cuda.is_available()

    # Instantiate model
    gan = MNISTWGAN(args.data_root, args.debug, cuda_enabled, args.quiet)

    # Train model
    gan.train()

    # Save model
    if args.save_model:
        torch.save(gan.g.state_dict(), 'mnist_gen.pt')

