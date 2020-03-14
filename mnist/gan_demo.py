import torch
import streamlit as st
from skimage.transform import rescale

from train_gan import G

st.title('MNIST GAN Demo')

# Load generator
gen = G(100, 28*28)
sd = torch.load('mnist_gen.pt')
gen.load_state_dict(sd)
gen = gen.cuda()

if st.button('Generate digit'):
    # Create latent vector
    z = torch.FloatTensor(1, 100).normal_()

    # Check for cuda
    if torch.cuda.is_available():
        z = z.cuda()

    # Generate image
    image = gen(z)
    
    # Move from GPU to CPU
    image = image.cpu().detach().numpy()

    # Unflatten
    image = image.reshape(28, 28)

    # Rescale image from -1,1 to 0,255
    image = image*128 + 128

    # Increase image size
    image = rescale(image, 10)
    
    # Write image to interface
    st.image(image, clamp=True)

