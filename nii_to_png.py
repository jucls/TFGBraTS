import nibabel as nii
import numpy as np
from pathlib import Path
from PIL import Image


def nii_to_png(nii_path: Path, imgs_out_path: Path):
    # read nii file
    data = nii.load(nii_path).get_fdata()
    # rotate the image by 90 degrees so it is horizontal
    data = np.rot90(data, axes=(0, 1))
    # reading all the images in the nii, we want to iterate on the z (2) dimension
    for i in range(data.shape[2]):
        img = data[..., i]
        # image normalization
        max_val = np.max(img)
        if max_val != 0:
            img = (img / max_val) * 255
        # saving the image
        img = Image.fromarray(img.astype(np.uint8))
        name = Path(nii_path.stem + "_S" + str(i) + ".png")
        img.save(imgs_out_path / name, optimize=True)

