import sys
import os
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QSpinBox, QFormLayout, QToolBar, QAction, QErrorMessage, QDialog, QSplashScreen, QDesktopWidget, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPainter, QColor
from PyQt5.QtCore import Qt, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
from predictions import clasify, segment

class PointCloud:
    def __init__(self, nii_data):
        x, y, z = np.nonzero(nii_data)
        self.points = np.column_stack([x, y, z]).astype(np.float32)
        self.values = nii_data[x, y, z].astype(np.float32)

    def get_points(self):
        return self.points
    
    def get_values(self):
        return self.values

class InfoWindow(QDialog):
    def __init__(self, classification_info, parent=None):
        super(InfoWindow, self).__init__(parent)
        
        self.setWindowTitle("Resultados de Clasificación")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Agregar la información de clasificación a la ventana
        info_label = QLabel(f"Tipo de Tumor: {classification_info}")
        layout.addWidget(info_label)

        # Botón para cerrar la ventana
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.setLayout(layout)

        self.setStyleSheet("""
            QDialog {
                background-color: #2E3440;
                color: #D8DEE9;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel {
                font-size: 16px;
                color: #D8DEE9;
            }
            QPushButton {
                background-color: #4C566A;
                color: #D8DEE9;
                border: none;
                padding: 5px 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QPushButton:pressed {
                background-color: #81A1C1;
            }
        """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Brain Tumor Segmentator")
        self.setGeometry(100, 100, 1200, 800)
        self.center()

        # Establecer el ícono de la ventana
        icon_path = os.path.join(os.path.dirname(__file__), 'GUI', 'resources', 'icons', 'icono.png')
        self.setWindowIcon(QIcon(icon_path))

        self.nii_data = None
        self.nii_name = None
        self.nii_seg = None
        self.initUI()
    
    def center(self):
        # Método para centrar la ventana en la pantalla
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def initUI(self):
        # Crear una barra de herramientas superior
        toolbar = QToolBar("Barra principal")
        self.addToolBar(toolbar)

        generate_action = QAction("Generar Puntos", self) # Se hace automático al cargarlo...
        generate_action.triggered.connect(self.generate_points)
        toolbar.addAction(generate_action)

        load_action = QAction("Cargar", self)
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)

        clasifica = QAction("Clasificar", self)
        clasifica.triggered.connect(self.clasifica)
        toolbar.addAction(clasifica)

        segmentacion = QAction("Auto-segmentación", self)
        segmentacion.triggered.connect(self.segmenta)
        toolbar.addAction(segmentacion)

        editor = QAction("Editor", self)
        editor.triggered.connect(self.edita)
        toolbar.addAction(editor)

        exportacion = QAction("Exportar", self)
        exportacion.triggered.connect(self.exporta)
        toolbar.addAction(exportacion)

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

        # Crear contenedores negros para las imágenes y el renderizador
        self.axial_container = self.create_black_container()
        self.coronal_container = self.create_black_container()
        self.sagittal_container = self.create_black_container()

        # Añadir los contenedores al layout de la cuadrícula
        grid_layout.addWidget(self.axial_container, 0, 0)
        grid_layout.addWidget(self.coronal_container, 0, 1)
        grid_layout.addWidget(self.sagittal_container, 1, 0)

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
    
    def create_black_container(self):
        container = QWidget()
        container.setStyleSheet("""
            background-color: black;
            border-radius: 15px;
            """)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        container.setLayout(layout)
        return container

    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.nii_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo NII", "", "Archivos NII (*.nii *.nii.gz);;Todos los Archivos (*)", options=options)
        
        if self.nii_name:
            # Cargar el archivo NII
            nii_image = nib.load(self.nii_name)
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

            # Poner a None el atributo de segmentación
            self.nii_seg = None

            # Actualizar las imágenes
            self.update_slices()

            # Generar automáticamente la nube de puntos después de cargar el archivo
            self.generate_points()
            

    def update_slices(self):
        if self.nii_data is not None:
            axial_slice = self.nii_data[:, :, self.axial_spinbox.value()]
            coronal_slice = self.nii_data[:, self.coronal_spinbox.value(), :]
            sagittal_slice = self.nii_data[self.sagittal_spinbox.value(), :, :]

            self.display_image(axial_slice, self.axial_label, self.axial_spinbox.value(), 'axial')
            self.display_image(coronal_slice, self.coronal_label, self.coronal_spinbox.value(), 'coronal')
            self.display_image(sagittal_slice, self.sagittal_label, self.sagittal_spinbox.value(), 'sagittal')

    def display_image(self, slice_data, label, slice_index, slice_type):
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

        # Si existe segmentación y el índice del corte está dentro de los límites
        if self.nii_seg is not None and 0 <= slice_index < self.nii_seg.shape[0]:

            result_image = QImage(q_image.size(), QImage.Format_ARGB32)
            painter = QPainter(result_image)
            painter.drawImage(0, 0, q_image)

            # Obtener el corte de segmentación correspondiente al índice
            if slice_type == 'sagittal':
                seg_slice = self.nii_seg[:, slice_index, :]
                seg_slice = np.fliplr(np.rot90(np.rot90(seg_slice)))
            elif slice_type == 'coronal':
                seg_slice = self.nii_seg[:, :, slice_index]
                seg_slice = np.fliplr(np.rot90(np.rot90(seg_slice)))
            else:
                seg_slice = self.nii_seg[slice_index, :, :]
                seg_slice = np.rot90(seg_slice)  # Rotar la imagen 90 grados para visualizarla correctamente

            seg_image = QImage(seg_slice.shape[1], seg_slice.shape[0], QImage.Format_ARGB32)
            seg_image.fill(QColor(0, 0, 0, 0))  # Rellenar la imagen con color transparente

            # Dibujar los píxeles con valor 1 con opacidad
            for y in range(seg_slice.shape[0]):
                for x in range(seg_slice.shape[1]):
                    if seg_slice[y, x] == 1:
                        color = QColor(255, 0, 0, 128)  # Rojo con opacidad
                        seg_image.setPixelColor(x, y, color)
            # Superponer la segmentación en la imagen resultante
            painter.drawImage(0, 0, seg_image)
            painter.end()
            # Convertir QImage a QPixmap y mostrarla en el QLabel
            pixmap = QPixmap.fromImage(result_image)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
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
            self.plotter_widget.background_color = "black"
            self.plotter_widget.add_mesh(cloud, scalars="values", cmap="coolwarm", point_size=5, render_points_as_spheres=True, show_scalar_bar=False, opacity=0.5)

            # Ajustar la cámara
            self.plotter_widget.reset_camera()

            # Actualizar el render
            self.plotter_widget.update()
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("No se pueden generar volumen 3D porque no hay datos de resonancia magnética cargados.")

    def clasifica(self):
        if self.nii_data is not None and self.nii_data.size > 0:
            # Obtener la clasificación
            tag = clasify(self.nii_name)

            # Crear y mostrar la ventana de información
            info_window = InfoWindow(tag)
            info_window.exec_()
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("No se puede clasificar porque no hay datos de resonancia magnética cargados.")

    def segmenta(self):
        if self.nii_data is not None and self.nii_data.size > 0:
            # Introducir segmentación automática en array
            self.nii_seg = segment(self.nii_name)

            # Actualizar las imágenes
            self.update_slices()
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("No se puede segmentar porque no hay datos de resonancia magnética cargados.")

    def edita(self):
        if self.nii_data is not None and self.nii_data.size > 0:
            print("Código de editar")
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("Esta versión de interfaz requiere hacer auto-segmentación antes.")

    def exporta(self):
        if self.nii_seg is not None and self.nii_seg.size > 0:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Archivo NII", "", "Archivos NIfTI (*.nii.gz)", options=options)

            if file_path:
                try:
                    # Guardar el archivo NIfTI
                    nii_image = nib.Nifti1Image(self.nii_seg.astype(np.int16), np.eye(4))
                    nib.save(nii_image, file_path)
                    # Mostrar mensaje de éxito
                    message_box = QMessageBox()
                    message_box.setWindowTitle("Archivo exportado")
                    message_box.setText("El archivo se ha exportado satisfactoriamente.")
                    message_box.exec_()
                except Exception as e:
                    # Mostrar mensaje de error si ocurre alguna excepción
                    message_box = QMessageBox()
                    message_box.setWindowTitle("Error al Guardar")
                    message_box.setText(f"Se produjo un error al intentar exportar el archivo: {str(e)}")
                    message_box.exec_()
        else:
            error_message = QErrorMessage(self)
            error_message.showMessage("No se puede exportar segmentación sin antes realizar auto-segmentación.")

    def close_application(self):
        sys.exit()

def main():
    app = QApplication(sys.argv)

    splash_pix = QPixmap("GUI/resources/icons/icono.png")
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.FramelessWindowHint)
    splash.show()

    font = QFont("mada-700", 32, QFont.Bold) 
    splash.setFont(font)
    splash.showMessage("BraTS UGR", Qt.AlignBottom | Qt.AlignCenter, Qt.red)

    timer = QTimer()
    timer.singleShot(3000, splash.close)

    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()