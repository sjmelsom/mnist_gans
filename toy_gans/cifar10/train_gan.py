import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from skimage.io import imsave
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
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
    def __init__(self, experiment):
        # Hyperparams
        self.batch_size = 128
        self.epochs = 200
        self.z_dim = 100
        self.lr = 2e-4

        # Data augmentations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        # Instantiate data loaders
        self.train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.num_channels = 1 if self.train_dataset.data.ndim == 3 else self.train_dataset.data.shape[3]

        # Setup discriminator
        self.d = D(self.num_channels)
        self.d.cuda()

        # Setup generator
        self.g = G(self.z_dim, self.num_channels)
        self.g.cuda()

        # Setup loss and optimizers
        self.loss = nn.BCELoss()
        self.g_opt = optim.Adam(self.g.parameters(), lr=self.lr)
        self.d_opt = optim.Adam(self.d.parameters(), lr=self.lr)

        # Setup logging
        self.experiment = experiment
        exp_dir = os.path.join('experiments', 'gan', experiment)
        self.images_dir = os.path.join(exp_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def train_d(self, x):
        self.d.zero_grad()

        x_real = x.cuda()
        soft_labels = torch.FloatTensor(x.shape[0], 1).uniform_(0, 0.1).squeeze()
        soft_labels = soft_labels.cuda()

        # Pass real examples through discriminator
        d_output = self.d(x_real)

        # Compute discriminator loss for real examples
        real_loss = self.loss(d_output, soft_labels)
        
        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim, 1, 1).normal_()
        z = z.cuda()

        # Generate generated examples
        generated_images = self.g(z)
        generated_images = generated_images.cuda()

        # Create labels for generated examples
        soft_labels = torch.FloatTensor(self.batch_size, 1).uniform_(0.9, 1).squeeze()
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
        soft_labels = soft_labels.cuda()

        # Create latent noise vectors
        z = torch.FloatTensor(self.batch_size, self.z_dim, 1, 1).normal_()
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

        # For each epoch
        for epoch in t1:

            # For each batch
            for batch_index, (x, _) in tqdm(enumerate(self.train_loader), position=1, total=len(self.train_loader)):
                d_loss = self.train_d(x)

                # Get generated images and their corresponding labels
                generated_images, g_loss = self.train_g(x)

            # Save some example images
            self.log_images(generated_images, epoch)
            
            # Update progress bar
            t1.set_description(f'D: {d_loss:.2f} | G: {g_loss:.2f}')


    def log_images(self, generated_images, epoch):
        im_name = os.path.join(self.images_dir, f'epoch{epoch}.jpg')
        # imgs = generated_images.view(-1, 3, 32, 32)
        imgs = [img.cpu().detach().numpy() for img in generated_images]
        imgs = [rescale_intensity(img, out_range=(0, 255)).astype(np.uint8) for img in imgs]
        imgs = np.hstack(np.moveaxis(np.array(imgs), 1, 3))
        imsave(im_name, imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    args = parser.parse_args()
    gan = CIFAR10GAN(args.experiment)
    gan.train()

