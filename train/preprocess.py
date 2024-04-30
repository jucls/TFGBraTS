
"""
 Este programa convierte los archivos nii.gz a pt
 
"""

import nibabel as nib
import torch

print(torch.cuda.is_available())

ruta_importar = 'data/gliomas/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz'
mi_array_numpy = nib.load(ruta_importar).get_fdata()  

# Convertir el array de NumPy a tensor de PyTorch
tensor_pytorch = torch.tensor(mi_array_numpy)

# Especificar la ruta donde deseas guardar el archivo .pt
ruta_archivo_pt = 'gli00000.pt'

# Guardar el tensor de PyTorch en un archivo .pt
torch.save(tensor_pytorch, ruta_archivo_pt)
