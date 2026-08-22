import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
    
class HistoriaEntrenamiento:
    train_loss: list
    train_acc: list
    val_loss: list
    val_acc: list

def calcular_accuracy(y_pred, y_true):
    _, predicciones = torch.max(y_pred, 1)
    correctos = (predicciones == y_true).sum().item()
    return correctos / len(y_true)

def run_epoch(model, dataloader, criterion, optimizer=None, device='cpu'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_acc = 0.0
    
    context = torch.enable_grad() if is_train else torch.inference_mode()
    
    with context:
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item() * len(y)
            total_acc += calcular_accuracy(y_pred, y) * len(y)
            
    n = len(dataloader.dataset)
    return total_loss / n, total_acc / n

def entrenar(model, train_loader, val_loader, criterion, optimizer, device, max_epochs=10, patience=5, verbose=True):
    historia = HistoriaEntrenamiento([], [], [], [])
    
    mejor_val_loss = float('inf')
    epochs_sin_mejora = 0
    mejor_modelo = None
    
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        
        historia.train_loss.append(train_loss)
        historia.train_acc.append(train_acc)
        historia.val_loss.append(val_loss)
        historia.val_acc.append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch:02d}/{max_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            epochs_sin_mejora = 0
            import copy
            mejor_modelo = copy.deepcopy(model.state_dict())
        else:
            epochs_sin_mejora += 1
            if patience > 0 and epochs_sin_mejora >= patience:
                if verbose:
                    print(f"Early stopping en epoch {epoch}")
                break
                
    if mejor_modelo is not None:
        model.load_state_dict(mejor_modelo)
        
    return model, historia
