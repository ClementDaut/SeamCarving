import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#energy_basic() renvoie la matrice d'énergie de l'image, se basant sur les dérivées horizontales et verticales 
def energy_basic(img):
  # Initialisation  des matrices de convolution
  Cx = np.array([[-0.125, 0, 0.125], [-0.25, 0, 0.25], [-0.125, 0, 0.125]])
  Cy = np.array([[-0.125, -0.25, -0.125], [0, 0, 0], [0.125, 0.25, 0.125]])

  # Convolution
  energyx = convolve2d(img, Cx, mode='same')
  energyy = convolve2d(img, Cy, mode='same')
  energy = np.sqrt(energyx**2 + energyy**2)

  energy_min = np.min(energy)
  energy_max = np.max(energy)
  energy = (energy - energy_min) / (energy_max - energy_min)

  return energy

#creat_mat_energy_V() permet de calculer le chemin vertical de plus petite énergie de l'image
def creat_mat_energy_V(img):
    L, C = img.shape
    mat_energy = np.zeros((L, C))
    mat_energy[0] = img[0]

    for i in range(1, L):
        left = np.roll(mat_energy[i-1], 1)
        center = mat_energy[i-1]
        right = np.roll(mat_energy[i-1], -1)

        # gestion des bords
        left[0] = center[0]
        right[-1] = center[-1]

        mat_energy[i] = img[i] + np.minimum(np.minimum(left, center), right)

    return mat_energy

#creat_mat_energy_H() permet de calculer le chemin horizontal de plus petite énergie de l'image
def creat_mat_energy_H(img):
    L, C = img.shape
    mat_energy = np.zeros((L, C))
    mat_energy[:, 0] = img[:, 0]

    for j in range(1, C):
        up = np.roll(mat_energy[:, j-1], 1)
        center = mat_energy[:, j-1]
        down = np.roll(mat_energy[:, j-1], -1)

        # gestion des bords
        up[0] = center[0]
        down[-1] = center[-1]

        mat_energy[:, j] = img[:, j] + np.minimum(np.minimum(up, center), down)

    return mat_energy
