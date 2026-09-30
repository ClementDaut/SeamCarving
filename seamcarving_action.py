import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

import torch
from torchvision import models, transforms
from captum.attr import Saliency
from skimage.transform import resize

from seamcarving_energie import *
# cette fonction permet de trouver la seam de plus faible energie
def chemin_seam(mat_energy): 
  L = len(mat_energy)
  C = len(mat_energy[0])
  seam = np.zeros(L, dtype=int) # vecteur qui contient les indices des colonnes de la seam

  seam[L - 1] = np.argmin(mat_energy[L - 1]) # initialisation avec l'indice de la plus petite valeur de la derniere ligen de la matrice d'énergie cumulée
  min_energy = min(mat_energy[L-1]) # on retient l'énergie de la seam qui sera supprimée

  for i in range(L - 2, -1, -1):
    j = seam[i + 1] # on se place à la colonne du haut de la seam

    if j == 0: # effet de bord
      indice = [j, j + 1]
    elif j == C - 1: # effet de bord
      indice = [j - 1, j]
    else: # tous les autres cas
      indice = [j - 1, j, j + 1]

    min_indice = np.argmin(mat_energy[i, indice]) # on regarde de quel est la valeur la plus faible du vecteur énegie
    seam[i] = indice[min_indice] # et on actualise seam avec l'indice de la nouvelle colonne

  return seam, min_energy

# cette fonction permet de supprimer une seam
def supprimer_seam(img, seam): 
  L = len(img)
  C = len(img[0])
  new_img = np.zeros((L, C - 1, 3), dtype=img.dtype)

  for i in range(L):
    j = seam[i] # Pour chaque ligne on regarde l'indice de la colonne du pixel à suppriner

    # Pixels à gauche de la seam
    new_img[i, :j] = img[i, :j]
    # Pixels à droite de la seam
    new_img[i, j:] = img[i, j+1:] # On saute l'indice j qui correspond au pixel à supprimer

  return new_img

# cette fonction permet de dupliquer une seam
def ajouter_seam(img, seam, offset): 
  L = len(img)
  C = len(img[0])
  new_img = np.zeros((L, C + 1, 3), dtype=img.dtype)

  for i in range(L):
    j = min(seam[i] + offset, C - 1) #comme on ajoute toutes les seam à la suite on met un offset pour gérere l'augmentation de la taille de l'image

    # Pixels à gauche de la seam
    new_img[i, :j] = img[i, :j]
    # Pixel qui doit être dupliqué
    new_img[i, j] = img[i,j] #le pixel de la colonne j est dupliqué
    # Pixels à droite de la seam
    new_img[i, j + 1:] = img[i, j:]

  return new_img

# Cette fonction détermine toutes les seam de plus faibles énergies
def toutes_seams(img, nb_pix): 
    seams = [] #liste des seams de plus faibles énergies
    img_copy = img.copy()

    for i in range(nb_pix):
      img_gray = np.mean(img_copy, axis=2) # On met l'image en gris
      energy = energy_basic(img_gray) # On calcule la dérivée de l'image
      mat_energy = creat_mat_energy_V(energy) # On calcule la matrice d'énergie cumulée
      new_seam, min_energy = chemin_seam(mat_energy) # On calcule la seam de plus faible énergie et on stocke cette énergie
      seams.append(new_seam) # On ajoute cette seam à la liste des seams
      img_copy = supprimer_seam(img_copy, new_seam) # On supprime la seam et on recommence

    return seams

# Cette fonction duplique toutes les seams de plus faible énergie
def inserer_seams(img, seams): 
  L = len(img)
  C = len(img[0])
  offset = 0 # Création de l'offset
  new_img = img.copy()

  # réorganisation des seams qui vont être dupliquée
  # on duplique les seams de gauche à droite
  seams_sorted = sorted(enumerate(seams), key=lambda x: [x[1][i] + i * len(seams) for i in range(len(x[1]))])
  seams_ordered = [s for _, s in seams_sorted]

  for seam in seams_ordered:
    new_img = ajouter_seam(new_img, seam, offset)
    offset += 1  # incrémentation de l'offset


  return new_img

# Fonction qui permet d'agrandire l'image dans la largeur
def agrandir_V(image, nb_pix): 
  img = image.copy()

  seams = toutes_seams(img, nb_pix) # Trouver les seams de plus faible énergie
  img_agrandie = inserer_seams(img, seams) # Insérer tous les seams simultanément

  return img_agrandie

# Fonction qui permet d'agrandire l'image dans la hauteur
def agrandir_H(image, nb_pix): 
  # Transposer pour appliquer l'agrandissement vertical sur les lignes
  img_T = np.transpose(image, (1, 0, 2))  # (L, C, 3) -> (C, L, 3)

  seams = toutes_seams(img_T, nb_pix) # Trouver les seams de plus faible énergie
  img_agrandie_T = inserer_seams(img_T, seams) # Insérer tous les seams dans l'image transposée

  # Re-transposer pour retrouver la bonne image
  img_agrandie = np.transpose(img_agrandie_T, (1, 0, 2))  # (C + nb_pix, L, 3) -> (L + nb_pix, C, 3)

  return img_agrandie

# dessiner_seams() permet de visualiser les seams de moindre énergie choisies sur l'image
def dessiner_seams(image, seams, couleur=(255, 0, 0)):
    img_visu = image.copy()
    for seam in seams:
        for i in range(len(seam)):
            j = seam[i]
            if 0 <= i < img_visu.shape[0] and 0 <= j < img_visu.shape[1]:
                img_visu[i, j] = couleur
    return img_visu

# reduction_V permet de réduire l'image dans la largeur
def reduction_V(image, nb_pix):
  img = image.copy()
  energy_suppr = 0

  for i in range(nb_pix):
    img_gray = np.mean(img, axis=2) # On met l'image en gris
    energy = energy_basic(img_gray) # On calcule la dérivée de l'image
    mat_energy = creat_mat_energy_V(energy) # On calcule la matrice d'énergie cumulée
    seam, min_energy = chemin_seam(mat_energy) # On calcule la seam de plus faible énergie et on stocke cette énergie
    energy_suppr = energy_suppr + min_energy # On actualise l'énergie total supprimée
    img = supprimer_seam(img, seam) # On supprime la seam et on recommence

  return img, energy_suppr

# reduction_H() permet de réduire l'image dans la hauteur
def reduction_H(image, nb_pix): 
  img = np.transpose(image, (1, 0, 2))  # On transpose pour travailler verticalement
  energy_suppr = 0

  for i in range(nb_pix):
    img_gray = np.mean(img, axis=2) # On met l'image en gris
    energy = energy_basic(img_gray) # On calcule la dérivée de l'image
    mat_energy = creat_mat_energy_V(energy) # On calcule la matrice d'énergie cumulée
    seam, min_energy = chemin_seam(mat_energy) # On calcule la seam de plus faible énergie et on stocke cette énergie
    energy_suppr = energy_suppr + min_energy # On actualise l'énergie total supprimée
    img = supprimer_seam(img, seam) # On supprime la seam et on recommence

  img_out = np.transpose(img, (1, 0, 2))  # Re-transpose
  return img_out, energy_suppr

# removing_V() permet de supprimer une ligne dans l'image 
def removing_V(energy, image):
  seam, min_energy = chemin_seam(energy)
  img = supprimer_seam(image, seam)
  return img, min_energy

# removing_H() permet de supprimer une ligne dans l'image 
def removing_H(energy, image):
    # Transposer pour travailler comme si c'était vertical
    energy_T = energy.T
    img_T = np.transpose(image, (1, 0, 2))  # pour passer de (H, L, 3) à (L, H, 3)

    seam, min_energy = chemin_seam(energy_T)
    img_reduced_T = supprimer_seam(img_T, seam)

    # Re-transposer pour la remetre dans le bon sens
    img_reduced = np.transpose(img_reduced_T, (1, 0, 2))
    return img_reduced, min_energy

#Fonction générée par ChatGPT permettant de déterminer la saillance des images en utilisant des réseaux de neuronnes
def saillance_comparaison(image_c, image_reduite):
    #Charger modèle pré-entraîné
    model = models.resnet50(pretrained=True)
    model.eval()

    #Fonction de prétraitement
    def preprocess_tensor(img_np):
        img = np.stack([img_np] * 3, axis=-1) if img_np.ndim == 2 else img_np
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        img = transform(img)
        return img.unsqueeze(0)

    def enhance_contrast(saliency_map, gamma=0.3):
        saliency_map = np.clip(saliency_map, 0, None)  # Assure que les valeurs sont non négatives
        saliency_map = saliency_map / (saliency_map.max() + 1e-8) # Normalisation pour éviter division par zéro
        return np.power(saliency_map, gamma)

    #Générer image compressée par resize
    image_resized = (resize(image_c, image_reduite.shape, mode='reflect', anti_aliasing=True)*255).astype(np.uint8)

    #Prétraitement
    input_c = preprocess_tensor(image_c)
    input_r = preprocess_tensor(image_reduite)
    input_resized = preprocess_tensor(image_resized)
    input_c.requires_grad = True
    input_r.requires_grad = True
    input_resized.requires_grad = True

    #Cartes de saillance
    saliency = Saliency(model)
    target_c = model(input_c).argmax()
    target_r = model(input_r).argmax()
    target_resize = model(input_resized).argmax()

    saliency_c = saliency.attribute(input_c, target=target_c).squeeze().abs().detach().numpy()
    saliency_r = saliency.attribute(input_r, target=target_r).squeeze().abs().detach().numpy()
    saliency_resized = saliency.attribute(input_resized, target=target_resize).squeeze().abs().detach().numpy()

    saliency_c = np.max(saliency_c, axis=0)
    saliency_r = np.max(saliency_r, axis=0)
    saliency_resized = np.max(saliency_resized, axis=0)

    saliency_c_resized = resize(saliency_c, saliency_r.shape, mode='reflect', anti_aliasing=True)

    #Contraste pour affichage
    saliency_c_viz_full = enhance_contrast(saliency_c)
    saliency_c_viz = enhance_contrast(saliency_c_resized)
    saliency_r_viz = enhance_contrast(saliency_r)
    saliency_resized_viz = enhance_contrast(saliency_resized)

    #Scores
    score_carved = np.abs(saliency_c_resized - saliency_r).mean()
    score_resized = np.abs(saliency_c_resized - saliency_resized).mean()

    print(f"Diff. moyenne de saillance (carving) : {score_carved:.5f}")
    print(f"Diff. moyenne de saillance (resize)  : {score_resized:.5f}")

    #Affichage
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(saliency_c_viz_full, cmap='inferno_r')
    plt.title("Saillance Originale")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(saliency_c_viz, cmap='inferno_r')
    plt.title("Saillance Originale (resized)")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(saliency_r_viz, cmap='inferno_r')
    plt.title("Saillance Carvée")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(saliency_resized_viz, cmap='inferno_r')
    plt.title("Saillance Resize Classique")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

#T() renvoie l'ordre précis des seams à enlever
def T(r,c,img,img_c,mat_energy):

  mat_image_reduite = np.empty((r, c), dtype=object) #contient l'image réduite pour chaque suppression de seam (tableau de tableau)
  mat_image_reduite[0,0] = img_c

  mat_T = np.zeros((r,c))
  for i in range(1,c):
    img_prec_c = mat_image_reduite[0,i-1]
    img_prec = np.mean(img_prec_c, axis=2)
    mat_energy = creat_mat_energy_V(energy_basic(img_prec))
    img_reduite,E_sx = removing_V(mat_energy, img_prec_c) # On récupere la valeur de l'energie pour enlever la meilleure colonne et l'image sans cette colonne

    #MAJ matrice contenant énergie + matrice contenant images
    mat_image_reduite[0,i] = img_reduite
    mat_T[0,i] = mat_T[0,i-1] + E_sx

  for j in range(1,r):
    img_prec_c = mat_image_reduite[j-1,0]
    img_prec = np.mean(img_prec_c, axis=2)
    mat_energy = creat_mat_energy_H(energy_basic(img_prec))
    img_reduite,E_sy = removing_H(mat_energy, img_prec_c) # On récupere la valeur de l'energie pour enlever la meilleure ligne et l'image sans cette ligne

    #MAJ maatrice contenant énergie + matrice contenant images
    mat_image_reduite[j,0] = img_reduite
    mat_T[j,0] = mat_T[j-1,0] + E_sy

  #A ce stade on a rempli toute la premiere ligne et toute la premiere colonne de la matrice

  pos_x = 1
  pos_y = 1
  while pos_x < r or pos_y < c:

    if pos_y < c:

      for i in range(pos_y,c): # On remplit une ligne
        mat_a_c = mat_image_reduite[pos_x-1,i]
        mat_a = np.mean(mat_a_c, axis=2)
        e_a = energy_basic(mat_a)
        mat_b_c = mat_image_reduite[pos_x,i-1]
        mat_b = np.mean(mat_b_c, axis=2)
        e_b = energy_basic(mat_b)
        a = mat_T[pos_x-1,i] + removing_H(creat_mat_energy_H(e_a),mat_a_c)[1]
        b = mat_T[pos_x,i-1] + removing_V(creat_mat_energy_V(e_b),mat_b_c)[1]

        if a < b:
          mat_T[pos_x,i] = a
          mat_image_reduite[pos_x,i] = removing_H(creat_mat_energy_H(e_a),mat_a_c)[0]
        else:
          mat_T[pos_x,i] = b
          mat_image_reduite[pos_x,i] = removing_V(creat_mat_energy_V(e_b),mat_b_c)[0]

    if pos_x < r:

      for j in range(pos_x,r): # On remplit une colonne
        mat_a_c = mat_image_reduite[j-1,pos_y]
        mat_a = np.mean(mat_a_c, axis=2)
        e_a = energy_basic(mat_a)
        mat_b_c = mat_image_reduite[j,pos_y-1]
        mat_b = np.mean(mat_b_c, axis=2)
        e_b = energy_basic(mat_b)
        a = mat_T[j-1,pos_y] + removing_H(creat_mat_energy_H(e_a),mat_a_c)[1]
        b = mat_T[j,pos_y-1] + removing_V(creat_mat_energy_V(e_b), mat_b_c )[1]

        if a < b:
          mat_T[j,pos_y] = a
          mat_image_reduite[j,pos_y] = removing_H(creat_mat_energy_H(e_a),mat_a_c)[0]
        else:
          mat_T[j,pos_y] = b
          mat_image_reduite[j,pos_y] = removing_V(creat_mat_energy_V(e_b), mat_b_c)[0]

    #MAJ point de départ
    if pos_x < r - 1:
      pos_x += 1
    if pos_y < c - 1:
      pos_y += 1

    # La matrice est maintenant remplie, il suffit de faire un backtracking partant d'en bas droite vers haut gauche en gardant en memoire le chemin emprunté

    chemin = [] #chemin contient uniquement des 0 ou des 1. 1 -> on enlève une colonne | 0 -> on enlève une ligne
    pos_x = r-1
    pos_y = c-1

    while pos_x > 0 and pos_y > 0:
      if mat_T[pos_x-1,pos_y] < mat_T[pos_x,pos_y-1]:
        chemin.append(0)
        pos_x -= 1
      else:
        chemin.append(1)
        pos_y -= 1
    #on sort du while quand on a atteint un bord
    if pos_x == 0:
      while pos_y > 0:
        chemin.append(1)
        pos_y -= 1
    else:
      while pos_x > 0:
        chemin.append(0)
        pos_x -= 1


    return chemin, mat_T[r-1,c-1]
  
#comparaison() compare l'energie totale enlevé de l'image par seam carving et par suppression directe des dernières seams de l'image
def comparaison(col_sup,lign_sup,img_c,img_nb):
    energie_r_cut = []
    energie_r_seamc = []

    img = img_nb
    copy_img_c = img_c
    L = len(img)
    C = len(img[0])
    Eb = energy_basic(img)

    energie_tot_sc = 0
    energie_tot_cut = 0
    energie_tot_T = 0

    chemin_opti = T(lign_sup,col_sup,img_nb,img_c,Eb)
    energie_tot_scT = []

    for i in range(1,col_sup):
        last_col = Eb[:,C-i]
        energie_tot_cut += sum(last_col)
        energie_r_cut.append(energie_tot_cut)

        img_reduite, energie = reduction_V(copy_img_c, 1)
        energie_tot_sc += energie
        energie_r_seamc.append(energie_tot_sc)
        copy_img_c = img_reduite

    copy_img_c = img_c
    for i in range(1,lign_sup):
        last_lin = Eb[L-i,:]
        energie_tot_cut += sum(last_lin)
        energie_r_cut.append(energie_tot_cut)

        img_reduite, energie = reduction_H(copy_img_c, 1)
        energie_tot_sc += energie
        energie_r_seamc.append(energie_tot_sc)
        copy_img_c = img_reduite

    copy_img_c = img_c
    for c in chemin_opti[0]:
        if c == 0:
            img_reduite, energie = reduction_H(copy_img_c, 1)
        else:
            img_reduite, energie = reduction_V(copy_img_c, 1)
        copy_img_c = img_reduite
        energie_tot_T += energie
        energie_tot_scT.append(energie_tot_T)

    plt.figure()
    plt.plot(energie_r_cut, label='Enlever les N dernières seam',c='b')
    plt.plot(energie_r_seamc, label='SeamCarving(N)',c='r')
    plt.plot(energie_tot_scT, label='SeamCarving(N) + T', c='m')
    plt.xlabel('Nombre de seam enlevées')
    plt.ylabel('Énergie totale enlevée')

    plt.figure()
    plt.plot(energie_r_seamc, label='SeamCarving(N)',c='r')
    plt.plot(energie_tot_scT, label='SeamCarving(N) + T', c='m')
    plt.xlabel('Nombre de seam enlevées')
    plt.ylabel('Énergie totale enlevée')

    plt.legend()
    plt.show()