# generate.py

import torch
from models.gan import Generator
import matplotlib.pyplot as plt

# Carregar gerador treinado
generator = Generator(latent_dim=50)
generator.load_state_dict(torch.load('generator_epoch_50.pth'))

generator.eval()

def generate_image(generator, latent_dim):
    z = torch.randn(1, latent_dim, 1, 1)
    generated_img = generator(z).detach().cpu()
    img = (generated_img + 1) / 2  # Reverter normalização
    plt.imshow(img.squeeze(0).permute(1, 2, 0))
    plt.show()

# Gerar uma imagem
generate_image(generator, latent_dim=50)
