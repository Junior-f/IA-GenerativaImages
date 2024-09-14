# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Importar as classes do gerador e discriminador
from models.gan import Generator, Discriminator

# Definir hiperparâmetros
latent_dim = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
num_epochs = 200

# Transformações para as imagens
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Carregar dataset CIFAR-10
dataset = CIFAR10(root='datasets/', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Inicializar modelos e otimizadores
generator = Generator(latent_dim)
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
loss_function = nn.BCELoss()

# Loop de treinamento
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Preparar dados
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)
        real_imgs = imgs
        z = torch.randn(imgs.size(0), latent_dim, 1, 1)
        fake_imgs = generator(z)

        # Treinar Discriminador
        optimizer_D.zero_grad()
        real_loss = loss_function(discriminator(real_imgs), real)
        fake_loss = loss_function(discriminator(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Treinar Gerador
        optimizer_G.zero_grad()
        g_loss = loss_function(discriminator(fake_imgs), real)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    # Salvar modelo a cada 50 épocas
    if epoch % 50 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch}.pth')
