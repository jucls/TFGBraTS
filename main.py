from nii_to_png import nii_to_png as unroll_nii
from pathlib import Path
import os


if __name__ == '__main__':

    dataset_src = Path("/home/jaime/Escritorio/TFG/niis")
    dataset_dst = Path("/home/jaime/Escritorio/TFG/imagenes")
    
    count = 0
    for path, dirs, files in os.walk(dataset_src):
        # we reached te data folder
        if not dirs:
            # creation of the folder that will contain the images
            dst_folder = Path(path.replace(str(dataset_src), str(dataset_dst)))
            try:
                dst_folder.mkdir(parents=True)
                print(dst_folder)
            except OSError as e:
                raise e
            
            for file in [f for f in files]:
                nii = Path(path) / Path(file)
                unroll_nii(nii, dst_folder)
                count += 1
                print(f"\t{file} was converted")
    
    print(f"{count} niis were converted.")

