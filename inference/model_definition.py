import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

# Dependencias de PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Dependencias de Fastai
from fastai.learner import Learner
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader
from fastai.vision.models import resnet34
from fastai.losses import L1LossFlat # MAE Loss for flatten (lineal)
from fastai.metrics import mae, BalancedAccuracy, accuracy
from fastai.test_utils import *
from fastai.vision.all import *


# Leer los dataFrames con las rutas y los datos
train = pd.read_csv('GUI/resources/partition/trainAtlas.csv')
valid = pd.read_csv('GUI/resources/partition/validAtlas.csv')

standard_normalize = lambda array: (array - np.mean(array)) / np.std(array) if np.std(array) != 0 else array - np.mean(array)

class BraTS(Dataset):
  """
    Permite la lectura de los datos y etiquetas.
    Combina ambos para formar un tensor el cual va a ser pasado a un dataloader.
  """
  def __init__(self, atlas, transform=None):
    """
      Constructor de la clase dataset.

      Args:
      atlas (DataFrame): Frame de datos que contiene las rutas a cada ejemplo, el nº de slice y su etiqueta.
      transform (bool, optional): Aplicar transformaciones a las imágenes. Por defecto, no.

    """
    super(BraTS, self).__init__()
    self.atlas = atlas
    self.transform = transform
    self.mri_dtype = torch.float32

  def __len__(self):
    """
      Devuelve el tamaño del dataset
    """
    return len(self.atlas)

  def __getitem__(self, index):

    """
     Devuelve un elemento del dataset

    """
    
    # Obtener los paths de cada imagen
    img_paths = self.atlas.iloc[index][['ruta_t1c', 'ruta_t1n', 'ruta_t2f']].tolist()
    slice_num = self.atlas.iloc[index]['slice']

    # Obtener las imágenes de cada prueba 
    X = np.zeros((240, 240, 3))
    for idx, img_path in enumerate(img_paths): 
      current_img = nib.load(img_path).dataobj[..., slice_num]       
      X[:, :, idx] += standard_normalize(current_img)
        
    x = torch.from_numpy(X.T).type(self.mri_dtype)
    y = torch.from_numpy(X.T).type(self.mri_dtype)
    
    del current_img
    del X
    return x, y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.sequence = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.sequence(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.sequence(x)
    
class UpConvResize(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvResize, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=8, stride=1)
        )

    def forward(self, x):
        return self.block(x)

class ResidualAutoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResidualAutoencoder, self).__init__()
        
        self.encoder = nn.ModuleList([
            *list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-2] # Tomar todas las capas excepto las últimas dos (avgpool y fc)
        ]) 

        self.bottleneck = ConvBlock(512, 1024)

        self.decoder = nn.ModuleList([
            UpConvResize(1024, 512), #15
            UpConv(512, 256), #30
            UpConv(256, 128), #60
            UpConv(128, 64), #120
        ])
        
        self.output = nn.Sequential(
            UpConv(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )      
        
    def forward(self, x):
        o = x       
        for layer in self.encoder:
            o = layer(o)
        o = self.bottleneck(o)

        for layer in self.decoder:
            o = layer(o)
            
        for layer in self.output:
            o = layer(o) 
            
        return o


class BraTS_BIN(Dataset):
    """
        Permite la lectura de los datos y etiquetas.
        Combina ambos para formar un tensor el cual va a ser pasado a un dataloader.
    """
def __init__(self, atlas, transform=None):
    """
    Constructor de la clase dataset.

    Args:
    atlas (DataFrame): Frame de datos que contiene las rutas a cada ejemplo, el nº de slice y su etiqueta.
    transform (bool, optional): Aplicar transformaciones a las imágenes. Por defecto, no.

    """
    super(BraTS_BIN, self).__init__()
    self.atlas = atlas
    self.transform = transform
    self.mri_dtype = torch.float32
    self.label_dtype = torch.float32

def __len__(self):
    """
    Devuelve el tamaño del dataset
    """
    return len(self.atlas)

def __getitem__(self, index):

    """
    Devuelve un elemento del dataset

    """
    
    # Obtener los path de cada imagen
    img_paths = self.atlas.iloc[index][['ruta_t1c', 'ruta_t1n', 'ruta_t2f']].tolist()
    label = self.atlas.iloc[index]['etiqueta']
    slice_num = self.atlas.iloc[index]['slice']

    brain_img = np.zeros((240, 240, 3))
    
    for idx, img_path in enumerate(img_paths): # Combinar cada slice de las 3 pruebas
        current_img = nib.load(img_path).dataobj[..., slice_num]  
        brain_img[:, :, idx] += standard_normalize(current_img)
        
    x = torch.from_numpy(brain_img.T).type(self.mri_dtype)
    y = torch.tensor(label, dtype=torch.long)

    del brain_img
    del current_img
    return x, y

class BinaryNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, encoder_model=None):
        super(BinaryNet, self).__init__()
        
        # Encoder
        self.encoder = encoder_model.encoder
        
        # Representación latente
        self.bottleneck = encoder_model.bottleneck
        
        self.flatten = nn.Flatten()
        
        # Capas densamente conectadas
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 8 * 8, 512),  # Capa lineal con 1024*8*8 entradas y 512 salidas
            nn.ReLU(),
            nn.Linear(512, 256),            # Capa lineal con 512 entradas y 256 salidas
            nn.ReLU(),
            nn.Linear(256, 128),            # Capa lineal con 256 entradas y 128 salidas
            nn.ReLU(),
            nn.Linear(128, 3)              # Capa lineal con 128 entradas y 3 salidas
        )
        
    def forward(self, x):
        o = x 
        for layer in self.encoder:
            o = layer(o)

        o = self.bottleneck(o)
        o = self.flatten(o)
        o = self.classifier(o)   
        return o

class BraTS_SEG(Dataset):
  """
    Permite la lectura de los datos y etiquetas.
    Combina ambos para formar un tensor el cual va a ser pasado a un dataloader.
  """
  def __init__(self, atlas, transform=None):
    """
      Constructor de la clase dataset.

      Args:
      atlas (DataFrame): Frame de datos que contiene las rutas a cada ejemplo, el nº de slice y su etiqueta.
      transform (bool, optional): Aplicar transformaciones a las imágenes. Por defecto, no.

    """
    super(BraTS_SEG, self).__init__()
    self.atlas = atlas
    self.transform = transform
    self.mri_dtype = torch.float32
    self.label_dtype = torch.float32

  def __len__(self):
    """
      Devuelve el tamaño del dataset
    """
    return len(self.atlas)

  def __getitem__(self, index):

    """
     Devuelve un elemento del dataset

    """
    
    # Obtener los path de cada imagen
    img_paths = self.atlas.iloc[index][['ruta_t1c', 'ruta_t1n', 'ruta_t2f']].tolist()
    slice_num = self.atlas.iloc[index]['slice']

    brain_img = np.zeros((240, 240, 3))
    
    for idx, img_path in enumerate(img_paths): # Combinar cada slice de las 3 pruebas
      current_img = nib.load(img_path).dataobj[..., slice_num]  
      brain_img[:, :, idx] += standard_normalize(current_img)

    # Obtener label
    label_path = img_paths[0].replace('t1c', 'seg')
    label = nib.load(label_path).dataobj[..., slice_num]
    
    # Binarizarlo mediante regla 'Whole Tumor'
    label_img = np.where(label != 0, 1, 0)
    
    x = torch.from_numpy(brain_img.T).type(self.mri_dtype)
    y = torch.from_numpy(label_img.T).type(self.mri_dtype)

    del brain_img
    del label
    del label_img
    del current_img

    return x, y

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, encoder_model=None):
        super(SegNet, self).__init__()
        
        # Encoder
        self.encoder = encoder_model.encoder
        
        # Representación latente
        self.bottleneck = encoder_model.bottleneck
    
        self.decoder = nn.ModuleList([
            UpConvResize(512+1024,512), #32
            UpConv(256+512,256), #64
            UpConv(128+256,128), #128
            UpConv(64+128,64) #256
        ])
        
        self.output = nn.Sequential(
            UpConv(64, 32),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )
        
        
    def forward(self, x):
        skips = []
        o = x
        for i, layer in enumerate(self.encoder):
            o = layer(o)
            skips.append(o)
        
        o = self.bottleneck(o)
        
        for i, layer in enumerate(self.decoder):
            o = torch.cat((skips[len(skips)-i-1],o), dim=1)
            o = layer(o)
            
        for layer in self.output:
            o = layer(o)
            
        return o

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice

def dice_loss(inputs, targets):
    loss = DiceLoss()
    return loss(inputs, targets)
    
def get_learner_classification():

    train_BS = BraTS(train)
    valid_BS = BraTS(valid)

    train_dl = DataLoader(train_BS, batch_size=32, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_BS, batch_size=32, pin_memory=True)

    autoencoder = ResidualAutoencoder()

    dls = DataLoaders(train_dl, valid_dl)
    learn = learn = Learner(
        dls=dls,  
        model=autoencoder,      
        loss_func=L1LossFlat(),    
        metrics=[mae]
    )

    model_path = Path('../models/fittingautoencodertotrain') # Por alguna razón me crea una carpeta y me tengo que salir de ella, .pth lo pone solo
    learn.load(model_path)

    train_BIN = BraTS_BIN() # Aunque se le pide un argumento no me lo pide, no preguntes. Y si se lo pongo me da error.
    valid_BIN = BraTS_BIN()

    train_dl = DataLoader(train_BIN, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_BIN, batch_size=32)
    binary_net = BinaryNet(encoder_model = learn.model)

    # Crear el objeto Learner
    balancedAccuracy = BalancedAccuracy()

    dls = DataLoaders(train_dl, valid_dl)
    learn_bin = Learner(
        dls=dls,  
        model=binary_net,      
        loss_func=FocalLossFlat(),    
        metrics=[accuracy, balancedAccuracy]
    )
    model_path = Path('../models/ternaryclassification') 
    learn_bin.load(model_path)
    return learn_bin

def get_learner_segmentation():

    train_BS = BraTS(train)
    valid_BS = BraTS(valid)

    train_dl = DataLoader(train_BS, batch_size=32, shuffle=True, pin_memory=True)
    valid_dl = DataLoader(valid_BS, batch_size=32, pin_memory=True)

    autoencoder = ResidualAutoencoder()

    dls = DataLoaders(train_dl, valid_dl)
    learn = learn = Learner(
        dls=dls,  
        model=autoencoder,      
        loss_func=L1LossFlat(),    
        metrics=[mae]
    )

    model_path = Path('../models/fittingautoencodertotrain') # Por alguna razón me crea una carpeta y me tengo que salir de ella, .pth lo pone solo
    learn.load(model_path)

    train_SEG = BraTS_SEG(train)
    valid_SEG = BraTS_SEG(valid)

    train_dl = DataLoader(train_SEG, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_SEG, batch_size=32)

    seg_net = SegNet(encoder_model = learn.model)

    dls = DataLoaders(train_dl, valid_dl)
    learn_seg = Learner(
        dls=dls,  
        model=seg_net,      
        loss_func=dice_loss,    
        metrics=[dice_loss]
    )

    model_path = Path('../models/segmentation')

    # Cargar los pesos
    learn_seg.load(model_path)
    return learn_seg