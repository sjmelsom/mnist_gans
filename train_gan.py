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
        self.train_dataset = datasets.MNIST(root='/data', train=True, download=True, transform=transform)
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
        exp_dir = os.path.join('experiments', experiment)
        self.images_dir = os.path.join(exp_dir, 'images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

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

        return d_loss.data.item()

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
        imgs = generated_images.view(-1, 28, 28)
        imgs = [img.cpu().detach().numpy() for img in imgs]
        imgs = [rescale_intensity(img, out_range=(0, 255)).astype(np.uint16) for img in imgs]
        imgs = np.hstack(imgs)
        imsave(im_name, imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment')
    args = parser.parse_args()
    gan = MNISTGAN(args.experiment)
    gan.train()

