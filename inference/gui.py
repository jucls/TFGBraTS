import tkinter as tk
from tkinter import filedialog
import re
from PIL import Image, ImageTk
import functions_gui as f_gui

def setTransparencia(valor):
    global nivel_transparencia_segmentos
    nivel_transparencia_segmentos = int(valor)

def ONSegmentacion():
    global segmentos
    segmentos = True

def OFFSegmentacion():
    global segmentos
    segmentos = False

def seleccionar_MRI():
    global path_seleccionado
    path_seleccionado = filedialog.askopenfilename(title="Seleccionar escáner", filetypes=[("Archivos NIfTI", "*.nii *.nii.gz")])
    # Cada imagen comienza en el 0
    setSlice(0, 'coronal')
    setSlice(0, 'axial')
    setSlice(0, 'sagital')

def setSlice(slice, vista):
    global segmentos, nivel_transparencia_segmentos
    # Si se ha abierto correctamente, mostrar inicio del archivo
    if path_seleccionado != '':
        datos = f_gui.loadMRI(path_seleccionado)
        seg = f_gui.loadMRI(re.sub(r'\b(t1c|t1n|t2f|t2w)\b', 'seg', path_seleccionado))

        if vista == 'axial':
            # Convertir a imagen PIL
            imagen = f_gui.combine(segmentos, seg[:, :, int(slice)], datos[:, :, int(slice)], nivel_transparencia_segmentos)
            imagen = Image.fromarray(imagen)
            # Mostrar imagen en Tkinter
            foto = ImageTk.PhotoImage(imagen)
            etiqueta_imagen_coronal.config(image=foto)
            etiqueta_imagen_coronal.image = foto  # Mantener una referencia para evitar que la imagen sea eliminada por el recolector de basura

        elif vista == 'coronal':
            # Convertir a imagen PIL
            imagen = f_gui.combine(segmentos, seg[int(slice), :, :], datos[int(slice), :, :], nivel_transparencia_segmentos)
            imagen = Image.fromarray(imagen).rotate(90)
            # Mostrar imagen en Tkinter
            foto = ImageTk.PhotoImage(imagen)
            etiqueta_imagen_axial.config(image=foto)
            etiqueta_imagen_axial.image = foto   
        else:
            # Convertir a imagen PIL
            imagen = f_gui.combine(segmentos, seg[:, int(slice), :], datos[:, int(slice), :], nivel_transparencia_segmentos)
            imagen = Image.fromarray(imagen).rotate(90)
            # Mostrar imagen en Tkinter
            foto = ImageTk.PhotoImage(imagen)
            etiqueta_imagen_sagital.config(image=foto)
            etiqueta_imagen_sagital.image = foto  


def main():

    global etiqueta_imagen_sagital
    global etiqueta_imagen_coronal
    global etiqueta_imagen_axial
    global path_seleccionado
    global segmentos
    global nivel_transparencia_segmentos

    ventana = tk.Tk()
    ventana.title("Buscador de datos en el datasets")
    ventana.geometry("800x800")
    path_seleccionado = ''
    segmentos = False
    nivel_transparencia_segmentos = 0

    # Crear un botón para seleccionar archivo
    boton_seleccionar = tk.Button(ventana, text="Seleccionar MRI", command=seleccionar_MRI)
    boton_seleccionar.grid(row=0, column=0)

    on_segmentacion= tk.Button(ventana, text="ON Segmentación", command=ONSegmentacion)
    on_segmentacion.grid(row=0, column=1)

    off_segmentacion= tk.Button(ventana, text="OFF Segmentación", command=OFFSegmentacion)
    off_segmentacion.grid(row=0, column=2)

    scale_slice_axial = tk.Scale(ventana, from_=0, to=154, orient=tk.VERTICAL, command=lambda valor: setSlice(valor, 'axial'))
    scale_slice_axial.grid(row=0, column=3)

    scale_slice_coronal = tk.Scale(ventana, from_=0, to=239, orient=tk.VERTICAL, command=lambda valor: setSlice(valor, 'coronal'))
    scale_slice_coronal.grid(row=0, column=4)

    scale_slice_sagital = tk.Scale(ventana, from_=0, to=239, orient=tk.VERTICAL, command=lambda valor: setSlice(valor, 'sagital'))
    scale_slice_sagital.grid(row=0, column=5)

    scale_transparencia = tk.Scale(ventana, from_=0, to=100, orient=tk.VERTICAL, command=lambda valor: setTransparencia(valor))
    scale_transparencia.grid(row=0, column=6)

    # Crear una etiqueta para mostrar la imagen
    etiqueta_imagen_sagital = tk.Label(ventana)
    etiqueta_imagen_sagital.grid(row=3, column=1)

    etiqueta_imagen_coronal = tk.Label(ventana)
    etiqueta_imagen_coronal.grid(row=2, column=1)

    etiqueta_imagen_axial = tk.Label(ventana)
    etiqueta_imagen_axial.grid(row=2, column=2)

    ventana.mainloop()

main()