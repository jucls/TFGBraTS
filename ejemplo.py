import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt


t1c_img1_nii = os.path.join(os.getcwd(), 'gliomas/BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz')
seg_img1_nii = os.path.join(os.getcwd(), 'gliomas/BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz')

t1c_img1 = nib.load(t1c_img1_nii)
seg_img1 = nib.load(seg_img1_nii)
# print(t1c_img1)
# print(seg_img1)

# img1_header = t1c_img1.header
# print(img1_header)

# Obtener el array de imagenes
image1_t1c = t1c_img1.get_fdata()
image1_t1c = np.rot90(image1_t1c, axes=(0, 1))

# Normalizar la imagen que queremos
slice = 150
imagen = image1_t1c[..., slice]
max_pixel= np.max(imagen)
if max_pixel != 0:
    imagen = (imagen / max_pixel) * 255.0

# Visualizar un slice
plt.imshow(imagen, cmap='gray')
plt.axis('off')  # Desactivar los ejes si no son necesarios
plt.show()