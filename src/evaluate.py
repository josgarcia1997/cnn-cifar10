import torch
import torch.nn.functional as F

def predecir(model, dataloader, device='cpu'):
    model.eval()
    todas_preds = []
    todas_reales = []
    todas_confianzas = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            probs = F.softmax(logits, dim=1)
            confianzas, preds = torch.max(probs, dim=1)
            
            todas_preds.extend(preds.cpu().numpy())
            todas_reales.extend(y.numpy())
            todas_confianzas.extend(confianzas.cpu().numpy())
            
    return todas_reales, todas_preds, todas_confianzas

def errores_alta_confianza(y_real, y_pred, confianzas, top_k=10):
    errores = []
    for i in range(len(y_real)):
        if y_real[i] != y_pred[i]:
            errores.append((i, y_real[i], y_pred[i], confianzas[i]))
            
    errores.sort(key=lambda x: x[3], reverse=True)
    return errores[:top_k]

def predict_image(image, model, device='cpu'):
    """
    Retorna la clase predicha, la probabilidad mayor y el top 3 de clases.
    """
    model.eval()
    with torch.inference_mode():
        image = image.unsqueeze(0).to(device)
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        
        prob_max, pred = torch.max(probs, dim=1)
        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)
        
    return pred.item(), prob_max.item(), top3_idx[0].cpu().numpy(), top3_probs[0].cpu().numpy()
