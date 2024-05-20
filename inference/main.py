import sys
import os
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QSpinBox, QFormLayout, QToolBar, QAction
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
import pyvista as pv
from pyvistaqt import QtInteractor

class PointCloud:
    def __init__(self, nii_data):
        x, y, z = np.nonzero(nii_data)
        self.points = np.column_stack([x, y, z]).astype(np.float32)
        self.values = nii_data[x, y, z].astype(np.float32)

    def get_points(self):
        return self.points
    
    def get_values(self):
        return self.values

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Brain Tumor Segmentator")
        self.setGeometry(100, 100, 1200, 800)

        # Establecer el ícono de la ventana
        icon_path = os.path.join(os.path.dirname(__file__), 'GUI', 'resources', 'icons', 'icono.png')
        self.setWindowIcon(QIcon(icon_path))

        self.nii_data = None
        self.initUI()

    def initUI(self):
        # Crear una barra de herramientas superior
        toolbar = QToolBar("Barra principal")
        self.addToolBar(toolbar)

        generate_action = QAction("Generar Puntos", self)
        generate_action.triggered.connect(self.generate_points)
        toolbar.addAction(generate_action)

        load_action = QAction("Cargar Archivo NII", self)
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)

        close_action = QAction("Cerrar Aplicación", self)
        close_action.triggered.connect(self.close_application)
        toolbar.addAction(close_action)

        # Widget principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal
        main_layout = QHBoxLayout()

        # Barra de opciones a la izquierda
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.axial_spinbox = QSpinBox(self)
        self.axial_spinbox.setEnabled(False)
        self.axial_spinbox.valueChanged.connect(self.update_slices)
        form_layout.addRow("Axial Slice:", self.axial_spinbox)

        self.coronal_spinbox = QSpinBox(self)
        self.coronal_spinbox.setEnabled(False)
        self.coronal_spinbox.valueChanged.connect(self.update_slices)
        form_layout.addRow("Coronal Slice:", self.coronal_spinbox)

        self.sagittal_spinbox = QSpinBox(self)
        self.sagittal_spinbox.setEnabled(False)
        self.sagittal_spinbox.valueChanged.connect(self.update_slices)
        form_layout.addRow("Sagittal Slice:", self.sagittal_spinbox)

        left_layout.addLayout(form_layout)
        left_widget.setLayout(left_layout)

        main_layout.addWidget(left_widget)

        # Layout para las imágenes en cuadrícula
        grid_layout = QGridLayout()

        # Configurar la expansión y alineación para que las imágenes y el QtInteractor ocupen todo el espacio disponible
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setAlignment(Qt.AlignCenter)

        self.axial_label = QLabel(self)
        self.axial_label.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.axial_label, 0, 0)

        self.coronal_label = QLabel(self)
        self.coronal_label.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.coronal_label, 0, 1)

        self.sagittal_label = QLabel(self)
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.sagittal_label, 1, 0)

        # Crear el widget de PyVista
        self.plotter_widget = QtInteractor(self)
        grid_layout.addWidget(self.plotter_widget.interactor)

        main_layout.addLayout(grid_layout)
        central_widget.setLayout(main_layout)

    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo NII", "", "Archivos NII (*.nii *.nii.gz);;Todos los Archivos (*)", options=options)
        
        if file_name:
            # Cargar el archivo NII
            nii_image = nib.load(file_name)
            self.nii_data = nii_image.get_fdata()

            # Habilitar y configurar los SpinBoxes
            self.axial_spinbox.setEnabled(True)
            self.axial_spinbox.setMaximum(self.nii_data.shape[2] - 1)
            self.axial_spinbox.setValue(self.nii_data.shape[2] // 2)

            self.coronal_spinbox.setEnabled(True)
            self.coronal_spinbox.setMaximum(self.nii_data.shape[1] - 1)
            self.coronal_spinbox.setValue(self.nii_data.shape[1] // 2)

            self.sagittal_spinbox.setEnabled(True)
            self.sagittal_spinbox.setMaximum(self.nii_data.shape[0] - 1)
            self.sagittal_spinbox.setValue(self.nii_data.shape[0] // 2)

            # Actualizar las imágenes
            self.update_slices()

            # Generar automáticamente la nube de puntos después de cargar el archivo
            self.generate_points()

    def update_slices(self):
        if self.nii_data is not None:
            axial_slice = self.nii_data[:, :, self.axial_spinbox.value()]
            coronal_slice = self.nii_data[:, self.coronal_spinbox.value(), :]
            sagittal_slice = self.nii_data[self.sagittal_spinbox.value(), :, :]

            self.display_image(axial_slice, self.axial_label)
            self.display_image(coronal_slice, self.coronal_label)
            self.display_image(sagittal_slice, self.sagittal_label)

    def display_image(self, slice_data, label):
        # Normalizar la imagen a la escala de grises de 0 a 255
        slice_data = np.rot90(slice_data)  # Rotar la imagen 90 grados para visualizarla correctamente
        max_val = np.max(slice_data)
        if max_val != 0:
            slice_data = (slice_data / max_val) * 255
        slice_data = slice_data.astype(np.uint8)

        # Convertir el array numpy a una imagen QImage
        height, width = slice_data.shape
        bytes_per_line = width
        q_image = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Convertir QImage a QPixmap y mostrarla en el QLabel
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def generate_points(self):
        if self.nii_data is not None and self.nii_data.size > 0:
            # Crear la nube de puntos
            point_cloud = PointCloud(self.nii_data)

            # Obtener los puntos y sus valores
            points = point_cloud.get_points()
            values = point_cloud.get_values()

            # Crear una nube de puntos en PyVista
            cloud = pv.PolyData(points)
            cloud["values"] = values

            # Limpiar y añadir la nube de puntos al widget
            self.plotter_widget.clear()
            self.plotter_widget.add_mesh(cloud, scalars="values", cmap="coolwarm", point_size=5, render_points_as_spheres=True)

            # Ajustar la cámara
            self.plotter_widget.reset_camera()

            # Actualizar el render
            self.plotter_widget.update()
        else:
            print("Error: No hay datos de resonancia magnética cargados.")
    
    def close_application(self):
        sys.exit()

def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()