from pathlib import Path

# Racine = dossier livrables/ (config.py est à la racine, au même niveau que les notebooks)
ROOT_DIR      = Path(__file__).resolve().parent

DATA_DIR      = ROOT_DIR / "data"
LABELED_DIR   = DATA_DIR / "avec_labels"
UNLABELED_DIR = DATA_DIR / "sans_label"
CANCER_DIR    = LABELED_DIR / "cancer"
NORMAL_DIR    = LABELED_DIR / "normal"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

FEATURES_DIR  = PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(exist_ok=True)

METADATA_PATH = PROCESSED_DIR / "metadata.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_CLUSTERS   = 2
RANDOM_STATE = 42

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
