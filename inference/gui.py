import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import functions_gui as f_gui

def seleccionar_MRI():
    global path_seleccionado
    path_seleccionado = filedialog.askopenfilename(title="Seleccionar escáner", filetypes=[("Archivos NIfTI", "*.nii *.nii.gz")])
    # Cada imagen comienza en el 0
    setSlice(0)

def setSlice(slice):
    # Si se ha abierto correctamente, mostrar inicio del archivo
    if path_seleccionado != '':
        datos = f_gui.loadMRI(path_seleccionado)
        # Convertir a imagen PIL
        imagen = Image.fromarray(datos[..., int(slice)])

        # Mostrar imagen en Tkinter
        foto = ImageTk.PhotoImage(imagen)
        etiqueta_imagen.config(image=foto)
        etiqueta_imagen.image = foto  # Mantener una referencia para evitar que la imagen sea eliminada por el recolector de basura 

def main():

    global etiqueta_imagen
    global path_seleccionado

    ventana = tk.Tk()
    ventana.title("Buscador de datos en el datasets")
    ventana.geometry("1000x800")
    path_seleccionado = ''

    # Crear un botón para seleccionar archivo
    boton_seleccionar = tk.Button(ventana, text="Seleccionar MRI", command=seleccionar_MRI)
    boton_seleccionar.pack(pady=10)

    scale_slice = tk.Scale(ventana, from_=0, to=154, orient=tk.VERTICAL, command=lambda valor: setSlice(valor))
    scale_slice.pack(side=tk.RIGHT, padx=10)

    # Crear una etiqueta para mostrar la imagen
    etiqueta_imagen = tk.Label(ventana)
    etiqueta_imagen.pack()

    ventana.mainloop()

main()