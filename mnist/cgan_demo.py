import torch
import numpy as np
import streamlit as st
from skimage.transform import rescale

from train_cgan import G

st.title('MNIST CGAN Demo')

# Load generator
gen = G(110, 28*28)
sd = torch.load('mnist_cgen.pt')
gen.load_state_dict(sd)
gen = gen.cuda()

# Text input
input_num = st.text_input('Enter digit (0-9)')

st.button('Generate')

# if st.button('Generate digit'):
if input_num != '':
    # Setup input digit
    num = torch.LongTensor(3, 1)
    num[:3] = int(input_num)

    # Check for cuda
    if torch.cuda.is_available():
        num = num.cuda()

    # Create latent vector
    z = torch.FloatTensor(3, 100).normal_()

    # Check for cuda
    if torch.cuda.is_available():
        z = z.cuda()

    # Generate image
    image = gen(z, num)
    
    # Move from GPU to CPU
    image = image.cpu().detach().numpy()

    # Unflatten
    image = np.hstack(image)

    # Rescale image from -1,1 to 0,255
    image = image*128 + 128

    # Increase image size
    image = rescale(image, 5)
    
    # Write image to interface
    st.image(image, clamp=True)

    # Draw balloons
    st.balloons()

