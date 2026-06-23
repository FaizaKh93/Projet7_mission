# BrainScanAI — Détection de tumeurs cérébrales

Projet de classification d'images IRM cérébrales (cancer / normal) combinant **transfer learning** (ResNet50), **apprentissage non supervisé** (clustering) et **apprentissage semi-supervisé** (weak labels + augmentation de données), pour exploiter un dataset où seule une faible fraction des images est labelisée.

## Contexte et objectif

Le dataset contient **1506 images IRM** (512×512, format `.jpg`), dont seulement **100 sont labelisées** (50 cancer / 50 normal) et **1406 sans label**. L'objectif est d'évaluer dans quelle mesure les 1406 images non labelisées peuvent être exploitées pour améliorer un classifieur entraîné sur très peu de données, via :
- le **clustering** (K-Means, DBSCAN) pour attribuer des labels faibles aux images non labelisées,
- l'**augmentation de données** sur les images labelisées (rotation, flip, zoom, luminosité, bruit) pour enrichir le jeu d'entraînement.

Trois modèles sont entraînés et comparés sur un même test set figé :
- **Modèle A** : supervisé pur (80 images fortement labelisées)
- **Modèle B** : semi-supervisé (1486 images, labels faibles DBSCAN + fine-tuning)
- **Modèle C** : semi-supervisé + augmentation (~2206 images, weak labels DBSCAN + 720 images augmentées + fine-tuning)

## Structure du projet

```
Projet7_mission/
├── data/
│   ├── avec_labels/
│   │   ├── cancer/                 # 50 images IRM labelisées "cancer"
│   │   └── normal/                 # 50 images IRM labelisées "normal"
│   ├── sans_label/                 # 1406 images IRM non labelisées
│   └── processed/                  # Fichiers produits par les notebooks (voir tableau ci-dessous)
│       └── features/               # Vecteurs de features ResNet50 (.npy) + métadonnées associées
├── notebooks/
│   ├── exploration_donnees.ipynb              # Étape 1
│   ├── preprocessing_feature_extraction.ipynb # Étape 2
│   ├── unsupervised_analysis.ipynb             # Étape 3
│   └── semi_supervised_learning.ipynb          # Étape 4
├── results/                        # Visualisations (.png) et poids de modèles (.pth)
├── src/
│   └── config.py                   # Chemins, hyperparamètres globaux et device (centralisés)
├── pyproject.toml                  # Dépendances du projet (gérées avec uv)
└── uv.lock
```

## Pipeline — détail des 4 notebooks

### Étape 1 — `exploration_donnees.ipynb`
Exploration initiale du dataset : chargement des chemins d'images, vérification de la résolution (512×512) et du mode couleur (RGB, bien que visuellement en niveaux de gris), statistiques de luminosité/contraste, visualisation d'exemples par classe.

- **Entrée** : `data/avec_labels/`, `data/sans_label/`
- **Sortie** : `data/processed/metadata.csv` (colonnes `path`, `label`, `split`)

### Étape 2 — `preprocessing_feature_extraction.ipynb`
Préparation des images et extraction de features par transfer learning.

1. **Split train/test** (effectué *avant toute transformation*, conformément aux bonnes pratiques de production) : les 100 images labelisées sont séparées en 80 train / 20 test (stratifié, 10 cancer + 10 normal en test). Le test set est sauvegardé immédiatement et n'est plus jamais touché par la suite.
2. **Preprocessing** : redimensionnement 224×224, normalisation avec les statistiques ImageNet.
3. **Extraction de features** : passage des 1506 images dans **ResNet50** pré-entraîné (poids ImageNet gelés, couche de classification retirée) → vecteur de **2048 features** par image.
4. **Augmentation des 80 images train labelisées** : 9 transformations appliquées à chacune (flip horizontal, rotations ±10°/±20°, zoom, luminosité ±20%, bruit gaussien) → 720 images augmentées supplémentaires, avec leurs vrais labels conservés. Les 20 images de test ne sont jamais augmentées.

- **Entrée** : `data/processed/metadata.csv`
- **Sorties** :
  - `data/processed/test_set.csv` — 20 images test (jamais utilisées avant l'évaluation finale)
  - `data/processed/features/resnet50_features.npy` — matrice (1506, 2048)
  - `data/processed/features/resnet50_metadata.csv` — métadonnées associées
  - `data/processed/features/resnet50_features_augmented.npy` — matrice (720, 2048)
  - `data/processed/features/resnet50_metadata_augmented.csv` — métadonnées associées (image d'origine, label, type d'augmentation)

### Étape 3 — `unsupervised_analysis.ipynb`
Analyse non supervisée des features ResNet50 pour générer des **labels faibles** sur les 1406 images sans label.

1. **Standardisation** (StandardScaler entraîné sans le test set) puis **réduction de dimensionnalité** : PCA (2048→150), puis UMAP et t-SNE pour la visualisation en 2D.
2. **Clustering** : K-Means (K=2) et DBSCAN, appliqués sur l'espace UMAP 2D, comparés via le score ARI (calculé uniquement sur les 80 images train labelisées) et le score de silhouette.
3. **Attribution des labels faibles** : pour chaque cluster, le label majoritaire parmi les images labelisées de ce cluster est propagé à toutes les images du cluster. Pour DBSCAN, les points de bruit (label -1) sont classés "cancer" par précaution (un faux positif est préférable à un faux négatif en contexte médical).
4. **DBSCAN retenu** comme méthode de labels faibles pour l'étape 4 (structure de clusters jugée plus naturelle que K-Means).

- **Entrée** : `data/processed/features/resnet50_features.npy`, `resnet50_metadata.csv`, `test_set.csv`
- **Sorties** :
  - `data/processed/clustering_comparison.csv` — comparaison K-Means vs DBSCAN (ARI, silhouette, points de bruit)
  - `data/processed/metadata_weak_labels_dbscan.csv` — 1506 images avec `weak_label_dbscan` et `cluster_dbscan`

### Étape 4 — `semi_supervised_learning.ipynb`
Entraînement et comparaison de 3 classifieurs MLP (PyTorch) sur les features ResNet50.

- **Architecture** : MLP 2048 → 512 → 128 → 2 (BatchNorm, Dropout, ReLU), ~1,1M paramètres.
- **Modèle A** : entraîné uniquement sur les 80 images fortement labelisées (baseline supervisée).
- **Modèle B** : pré-entraîné sur 1486 images (vrais labels + labels faibles DBSCAN), puis fine-tuné (learning rate réduit) sur les 80 images fortement labelisées.
- **Modèle C** : identique au modèle B, mais le pré-entraînement inclut en plus les 720 images augmentées (vrais labels) → ~2206 images en phase 1. Le fine-tuning reste sur les 80 images d'origine, pour isoler l'effet de l'augmentation en phase 1 et garder une comparaison équitable avec le modèle B.
- **Évaluation** sur le test set figé de 20 images, avec 3 métriques :
  - **Accuracy**
  - **F2-score** (macro) — poids du rappel doublé par rapport à la précision, pour réduire les faux négatifs (un cancer manqué est plus grave qu'une fausse alerte)
  - **ROC-AUC** — capacité de séparation des classes par probabilité, indépendamment du seuil de décision

- **Entrée** : `resnet50_features.npy`, `metadata_weak_labels_dbscan.csv`, `test_set.csv`, `resnet50_features_augmented.npy`, `resnet50_metadata_augmented.csv`
- **Sorties** :
  - `data/processed/model_comparison.csv` — tableau comparatif A/B/C (Accuracy, F2-score, ROC-AUC)
  - `results/confusion_matrices.png`, `results/learning_curves.png`
  - `results/model_A_supervised.pth`, `results/model_B_semi_supervised_dbscan.pth`, `results/model_C_semi_supervised_dbscan_aug.pth`

## Résultats (test set, 20 images)

| Modèle | Données d'entraînement | Accuracy | F2-score | ROC-AUC |
|---|---|---|---|---|
| A — Supervisé pur | 80 images (labels forts) | 0.90 | 0.8974 | 0.98 |
| B — Semi-sup. DBSCAN | 1486 images (DBSCAN weak labels) | 0.85 | 0.8440 | 0.94 |
| C — Semi-sup. DBSCAN + augmentation | ~2206 images | 0.90 | 0.8974 | 0.93 |

Le modèle supervisé pur (A) reste le plus performant sur ce test set. Les labels faibles DBSCAN seuls (B) dégradent légèrement la performance par rapport à A ; l'ajout de données augmentées (C) compense cette dégradation et ramène accuracy/F2-score au niveau de A, sans toutefois le dépasser. À noter : le test set ne comportant que 20 images, ces écarts restent sensibles à chaque image individuelle.

## Installation et exécution

Le projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion des dépendances.

```bash
uv sync
uv run jupyter lab
```

Les notebooks doivent être exécutés dans l'ordre (1 → 2 → 3 → 4), chaque étape dépendant des fichiers produits par la précédente.

## Configuration centralisée (`src/config.py`)

Tous les chemins (données brutes, fichiers traités, features, résultats) et hyperparamètres globaux (`RANDOM_STATE`, `N_CLUSTERS`, `DEVICE`) sont définis une seule fois dans `src/config.py` et importés par chaque notebook, afin d'éviter toute duplication ou incohérence entre les étapes.
