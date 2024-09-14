# utils.py

import torch

# Função para salvar checkpoints do modelo
def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

# Função para carregar checkpoints
def load_checkpoint(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
