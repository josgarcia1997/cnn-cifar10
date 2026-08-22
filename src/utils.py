import torch
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def detectar_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def contar_parametros(model):
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    totales = sum(p.numel() for p in model.parameters())
    return entrenables, totales
