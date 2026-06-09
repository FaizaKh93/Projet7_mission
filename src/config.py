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
