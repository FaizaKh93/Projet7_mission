# Ce fichier centralise tous les chemins du projet.
# En les définissant ici une seule fois, chaque notebook ou script
# peut les importer sans redéfinir les chemins manuellement.

from pathlib import Path

# Remonte d'un niveau depuis src/ pour atteindre la racine du projet
ROOT_DIR      = Path(__file__).resolve().parents[1]

# Dossier principal contenant toutes les images
DATA_DIR      = ROOT_DIR / "data" 

# Dossier des images dont le label (cancer / normal) est connu
LABELED_DIR   = DATA_DIR / "avec_labels"

# Dossier des images sans label — elles seront utilisées en semi-supervisé
UNLABELED_DIR = DATA_DIR / "sans_label"

# Sous-dossiers des images labellisées par classe
CANCER_DIR    = LABELED_DIR / "cancer"
NORMAL_DIR    = LABELED_DIR / "normal"

# Dossier des notebooks, utile si un script doit y écrire des résultats
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Dossier où seront sauvegardés les vecteurs de features extraits par le modèle pré-entraîné
FEATURES_DIR  = DATA_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)  # Crée le dossier s'il n'existe pas encore

# Fichier CSV contenant les métadonnées de toutes les images (path, label, split)
# Produit par l'étape 1, réutilisé par toutes les étapes suivantes
METADATA_PATH = DATA_DIR / "metadata.csv"

# Statistiques pixel calculées sur l'ensemble du dataset à l'étape 1
# Utilisées pour normaliser les images avant de les passer au modèle ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Moyenne par canal RGB (standard ImageNet)
IMAGENET_STD  = [0.229, 0.224, 0.225]  # Écart-type par canal RGB (standard ImageNet)

# Taille cible des images pour les modèles pré-entraînés (ResNet attend du 224x224)
IMG_SIZE = 224

# Dossier où seront sauvegardés les résultats du clustering (étape 3)
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Nombre de clusters cible -- correspond aux 2 classes connues : cancer et normal
N_CLUSTERS = 2

# Graine aleatoire pour reproductibilite -- meme valeur dans tous les notebooks
RANDOM_STATE = 42

# Appareil de calcul : GPU si disponible, sinon CPU
# torch.cuda.is_available() retourne True uniquement si un GPU NVIDIA avec CUDA est présent
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
