import numpy as np
import nibabel as nib

def loadMRI(path):
    img = nib.load(path).get_fdata()  # Cargar archivo NIfTI y obtener el array de la imagen
    imagen = np.rot90(img, axes=(0, 1))  # Rotar los datos para visualización  
    
    # Normalizar la imagen
    max_val = np.max(imagen)
    if max_val != 0:
        imagen = (imagen / max_val) * 255
    return imagen