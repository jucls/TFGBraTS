import numpy as np
import nibabel as nib

def loadMRI(path):
    imagen = nib.load(path).get_fdata()  # Cargar archivo NIfTI y obtener el array de la imagen 
    imagen = np.rot90(imagen, axes=(0, 1))  # Rotar los datos para visualización

    # Normalizar la imagen
    max_val = np.max(imagen)
    if max_val != 0:
        imagen = (imagen / max_val) * 255

    return imagen

def combine(segmentos, seg, imagen, transparencia):
    if (segmentos):

        # Iterar sobre cada píxel de la primera imagen y superponer en la segunda imagen si el píxel no es negro
        for i in range(seg.shape[0]):
            for j in range(seg.shape[1]):

                # Si el píxel no es negro
                if seg[i][j] > 0.0:
                    # Superponer el píxel de segmentación en la misma ubicación en la segunda imagen
                    imagen[i, j] = transparencia / 100

    return imagen