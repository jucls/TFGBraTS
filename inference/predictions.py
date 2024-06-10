import nibabel as nib
import re
import numpy as np
import torch
from model_definition import get_learner_classification, get_learner_segmentation, standard_normalize

def traerNII(ruta):
    """
     Dados tres paths de la resonancia, devuelve X las combinación de las tres resonancias.
     La entrada a los dos modelos.
    """
    paths = np.array([re.sub(r'\b(t1c|t1n|t2f|t2w)\b', 't1c', ruta), re.sub(r'\b(t1c|t1n|t2f|t2w)\b', 't1n', ruta), re.sub(r'\b(t1c|t1n|t2f|t2w)\b', 't2f', ruta)])
    t1c = nib.load(paths[0]).get_fdata()
    t1n = nib.load(paths[1]).get_fdata()
    t2f = nib.load(paths[2]).get_fdata()
    
    # Combinarlas
    X = np.zeros((155, 3, 240, 240))
    for i in range(155): # Combinar cada slice de las 3 pruebas en 3 canales  
      X[i, 0, :, :] += standard_normalize(t1c[:, :, i])
      X[i, 1, :, :] += standard_normalize(t1n[:, :, i])
      X[i, 2, :, :] += standard_normalize(t2f[:, :, i])
    
    return torch.from_numpy(X).type(torch.float32)


def clasify(ruta):
    """
     Dados tres paths de la resonancia, devuelve su clasificación.
     Importante: Esta función implementa el esquema de votación.
     return string "Meningioma" o "Glioma"
    """
    # Inferir todos los slices con la red
    X = traerNII(ruta)
    learn_bin = get_learner_classification()
    pred, _ = learn_bin.get_preds(dl=[(X,)])

    predictions = torch.argmax(pred, dim=1)
    conteo_meningioma = torch.sum(predictions == 0)
    
    # Si hay menos de 3 predicciones a meningioma, se devuelve glioma
    if conteo_meningioma < 3:
        return "Glioblastoma"
    else:
        return "Meningioma"

def segment(ruta):
    """
     Dados tres paths de la resonancia, devuelve su segmentación.
     return NumPy array (240 x 240 x 155) "0: Tejido sano" o "1: Tejido enfermo"
     Importante: Esta función implementa la umbralización de la salida de la red.
    """
    X = traerNII(ruta)
    learn_seg = get_learner_segmentation()
    pred, _ = learn_seg.get_preds(dl=[(X,)])
    pred = pred.squeeze(1)
    
    # Umbralizar la salida de la red.
    pred = torch.where(pred > 0.8, torch.tensor(1), torch.tensor(0))
    return pred.numpy()