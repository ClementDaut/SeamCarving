import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def affichage(image, color):

    print("Taille de l'image:",image.shape)
    if color == 1:                              #affichage en couleur
        plt.imshow(image)
    elif color == 0:                            #affichaeg en niveau de gris
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image.astype(int))           #affichage image réduite
    plt.axis('off')
    plt.show()



