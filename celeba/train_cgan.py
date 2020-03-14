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


class MNISTGAN:
    def __init__(self, experiment):
        # Hyperparams
        self.batch_size = 256
        self.epochs = 100
        self.z_dim = 100
        self.lr = 1e-4

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loader
        self.train_dataset = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

        num_digits = 10
        self.mnist_dim = self.train_dataset.train_data.size(1) * self.train_dataset.train_data.size(2)

        # Setup discriminator
        self.d = D(self.mnist_dim+num_digits, 1)
        self.d.cuda()

        # Setup generator
        self.g = G(self.z_dim+num_digits, self.mnist_dim)
        self.g.cuda()

        # Setup loss and optimizers
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)

        # Setup logging
        self.writer = SummaryWriter()
        self.experiment = experiment
        exp_dir = os.path.join('experiments', 'cgan', experiment)
        self.images_dir = os.path.join(exp_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def train_d(self, x, labels):
        self.d.zero_grad()

        x = x.cuda()
        labels = labels.long().cuda()

        # Pass real examples through discriminator
        d_output = self.d(x, labels)
        soft_labels = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1)
        soft_labels = soft_labels.cuda()

        # Compute discriminator loss for real examples
        real_loss = self.loss(d_output, soft_labels)
        
        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim).normal_()
        z = z.cuda()

        # Create labels
        labels = torch.FloatTensor(self.batch_size, 1).random_(0, 10)
        labels = labels.long().cuda()

        # Generate images
        generated_images = self.g(z, labels)

        # Pass generated examples through discriminator
        d_output = self.d(generated_images, labels)

        # Create labels for generated examples
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1)
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
        labels = torch.FloatTensor(self.batch_size, 1).random_(0, 10)
        labels = labels.long().cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(labels.shape[0], self.z_dim).normal_()
        z = z.cuda()

        # Pass noise and labels through generator to create generated images
        generated_images = self.g(z, labels)

        # Pass generated images and labels through discriminator
        d_output = self.d(generated_images, labels)

        # Create soft labels (near 0 or 1 instead of exactly 0 or 1)
        soft_labels = torch.FloatTensor(labels.shape[0], 1).uniform_(0, 0.1)
        # soft_labels = torch.FloatTensor(labels.shape[0], 1).uniform_(0.9, 1)
        soft_labels = soft_labels.cuda()

        # Compute how good the generated images look by how well they were able
        # to fool the discriminator.
        g_loss = self.loss(d_output, soft_labels)
        g_loss.cuda()

        # Step optimizer
        g_loss.backward()
        self.g_opt.step()

        return generated_images[:10], labels[:10], g_loss.data.item()


    def train(self):
        t1 = tqdm(range(self.epochs), position=0)

        # For each epoch
        for epoch in t1:

            # For each batch
            for batch_index, (x, y) in tqdm(enumerate(self.train_loader), position=1, total=len(self.train_loader)):
                d_loss = self.train_d(x, y)

                # Get generated images and their corresponding labels
                generated_images, labels, g_loss = self.train_g(x)

            self.writer.add_scalar('Generator Loss', g_loss, epoch)
            self.writer.add_scalar('Discriminator Loss', d_loss, epoch)
            imgs = make_grid(generated_images, normalize=True, scale_each=True)
            self.writer.add_image('Generated Images', imgs, epoch)

            # Save some example images
            # self.log_images(generated_images, labels, epoch)

            # Update progress bar
            t1.set_description(f'D: {d_loss:.2f} | G: {g_loss:.2f}')

        self.writer.close()

        # Just a little prompt as a demo
        while True:
            digit = input('Enter a number between 0 and 9: ')
            digit = torch.FloatTensor([[int(digit), 5]])
            digit = digit.long().cuda()
            z = torch.FloatTensor(2, self.z_dim).normal_()
            z = z.cuda()
            generated_image = self.g(z, digit)[0].cpu().detach().numpy()
            generated_image = rescale_intensity(generated_image, out_range=(0, 255)).astype(np.uint16)
            plt.imshow(generated_image)
            plt.show()


    def log_images(self, imgs, labels, epoch):
        labels = [label.cpu().detach().numpy() for label in labels]
        label = labels[0][0]
        im_name = os.path.join(self.images_dir, f'epoch{epoch}_digit{label}.png')
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

