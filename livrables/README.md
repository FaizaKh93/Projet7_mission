# BrainScanAI — Détection de tumeurs cérébrales

Projet de classification d'images IRM cérébrales (cancer / normal) combinant **transfer learning** (ResNet50), **apprentissage non supervisé** (clustering), **apprentissage semi-supervisé** (weak labels + augmentation de données) et **fine-tuning CNN** (dégel de layer4), pour exploiter un dataset où seule une faible fraction des images est labelisée.

---

## Utilisation du dossier `livrables/`

Le dossier `livrables/` est une version autonome du projet : tous les notebooks, `config.py`, et le dossier `data/` sont réunis à la racine. Il suffit de ce seul dossier pour exécuter l'intégralité du projet.

### Prérequis

Installer [uv](https://docs.astral.sh/uv/getting-started/installation/) si ce n'est pas déjà fait :

```bash
pip install uv
```

### Installation des dépendances

Depuis le dossier `livrables/` :

```bash
cd livrables/
uv sync
```

> **⚠ Configuration PyTorch** : le `pyproject.toml` fourni installe PyTorch pour **Windows + CUDA 12.1 + Python 3.11**. Si votre configuration est différente (Linux, Mac, pas de GPU, autre version de Python), supprimez le bloc `[tool.uv.sources]` dans `pyproject.toml` et relancez `uv sync` — uv choisira automatiquement la bonne version :
>
> ```toml
> # Supprimer ce bloc si votre config diffère de Windows/CUDA 12.1/Python 3.11
> [tool.uv.sources]
> torch = { url = "https://download.pytorch.org/whl/cu121/torch-2.2.2+cu121-cp311-cp311-win_amd64.whl" }
> torchvision = { url = "https://download.pytorch.org/whl/cu121/torchvision-0.17.2+cu121-cp311-cp311-win_amd64.whl" }
> ```

### Lancement de Jupyter

**Important** : Jupyter doit être lancé **depuis le dossier `livrables/`** pour que les chemins vers `config.py` et `data/` soient correctement résolus.

```bash
cd livrables/
uv run jupyter lab
```

Si Jupyter est lancé depuis un autre dossier (dossier parent par exemple), `config.py` ne sera pas trouvé et les notebooks échoueront à l'import.

### Ordre d'exécution des notebooks

Les notebooks s'exécutent dans l'ordre suivant, chaque étape produisant les fichiers nécessaires à la suivante :

```
1. exploration_donnees.ipynb              → génère data/processed/metadata.csv
2. preprocessing_feature_extraction.ipynb → génère features ResNet50 + test_set.csv
3. unsupervised_analysis.ipynb            → génère metadata_weak_labels_dbscan.csv
4. semi_supervised_learning_MLP.ipynb     → modèles MLP A/B/C (variante MLP)
5. semi_supervised_learning_CNN.ipynb     → modèles CNN A/B/C (variante CNN fine-tuning)
```

> **Note** : si les fichiers `data/processed/` sont déjà présents (cas où le dossier `livrables/` est fourni avec les données pré-traitées), les notebooks 2 à 5 peuvent être exécutés directement. Cependant, les chemins absolus stockés dans les CSV ont été générés sur la machine d'origine — **il est recommandé de ré-exécuter le notebook 1 en premier** pour régénérer les métadonnées avec les chemins corrects de votre machine.

---

## Contexte et objectif

Le dataset contient **1506 images IRM** (512×512, format `.jpg`), dont seulement **100 sont labelisées** (50 cancer / 50 normal) et **1406 sans label**. L'objectif est d'évaluer dans quelle mesure les 1406 images non labelisées peuvent être exploitées pour améliorer un classifieur entraîné sur très peu de données, via :
- le **clustering** (K-Means, DBSCAN) pour attribuer des labels faibles aux images non labelisées,
- l'**augmentation de données** sur les images labelisées (rotation, flip, zoom, luminosité, bruit) pour enrichir le jeu d'entraînement.

Trois modèles sont entraînés et comparés sur un même test set figé :
- **Modèle A** : supervisé pur (80 images fortement labelisées)
- **Modèle B** : semi-supervisé (1486 images, labels faibles DBSCAN + fine-tuning)
- **Modèle C** : semi-supervisé + augmentation (~2206 images, weak labels DBSCAN + 720 images augmentées + fine-tuning)

Deux variantes de classifieur sont comparées : **MLP** (sur features pré-extraites) et **CNN avec fine-tuning** (ResNet50 avec layer4 dégelé, entraîné directement sur les images).

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
│   ├── semi_supervised_learning_MLP.ipynb      # Étape 4 — variante MLP
│   └── semi_supervised_learning_CNN.ipynb      # Étape 5 — variante CNN (fine-tuning)
├── results/                        # Visualisations (.png) et poids de modèles (.pth)
├── src/
│   └── config.py                   # Chemins, hyperparamètres globaux et device (centralisés)
├── pyproject.toml                  # Dépendances du projet (gérées avec uv)
└── uv.lock
```

## Pipeline — détail des 5 notebooks

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
3. **Optimisation des HP DBSCAN** : recherche par grille sur `eps` ∈ {0.10, 0.15, 0.20, 0.25, 0.30} et `min_samples` ∈ {2, 3, 5}. Paramètres retenus : **eps=0.25, min_samples=3** (ARI=0.4200, 9 clusters, 1.1% de bruit — meilleur compromis entre ARI et compacité des clusters).
4. **Attribution des labels faibles** : pour chaque cluster, le label majoritaire parmi les images labelisées de ce cluster est propagé à toutes les images du cluster. Pour DBSCAN, les points de bruit (label -1) sont classés "cancer" par précaution (un faux positif est préférable à un faux négatif en contexte médical).
5. **DBSCAN retenu** comme méthode de labels faibles pour les étapes 4 et 5 (ARI=0.4200 > K-Means sur ce dataset).

- **Entrée** : `data/processed/features/resnet50_features.npy`, `resnet50_metadata.csv`, `test_set.csv`
- **Sorties** :
  - `data/processed/clustering_comparison.csv` — comparaison K-Means vs DBSCAN (ARI, silhouette, points de bruit)
  - `data/processed/metadata_weak_labels_dbscan.csv` — 1506 images avec `weak_label_dbscan` et `cluster_dbscan`

### Étape 4 — `semi_supervised_learning_MLP.ipynb` (variante MLP)
Entraînement et comparaison de 3 classifieurs MLP (PyTorch) sur les features ResNet50 pré-extraites.

- **Architecture** : MLP 2048 → 512 → 128 → 2 (BatchNorm, Dropout, ReLU), ~1,1M paramètres entraînables.
- **Modèle A** : entraîné uniquement sur les 80 images fortement labelisées (baseline supervisée).
- **Modèle B** : pré-entraîné sur 1486 images (vrais labels + labels faibles DBSCAN), puis fine-tuné (learning rate réduit) sur les 80 images fortement labelisées.
- **Modèle C** : identique au modèle B, mais le pré-entraînement inclut en plus les 720 images augmentées (vrais labels) → ~2206 images en phase 1. Le fine-tuning reste sur les 80 images d'origine, pour isoler l'effet de l'augmentation en phase 1 et garder une comparaison équitable avec le modèle B.
- **Évaluation** sur le test set figé de 20 images (Accuracy, F2-score macro β=2, ROC-AUC).

- **Entrée** : `resnet50_features.npy`, `metadata_weak_labels_dbscan.csv`, `test_set.csv`, `resnet50_features_augmented.npy`, `resnet50_metadata_augmented.csv`
- **Sorties** :
  - `data/processed/model_comparison.csv` — tableau comparatif A/B/C
  - `results/confusion_matrices.png`, `results/learning_curves.png`
  - `results/model_A_supervised.pth`, `results/model_B_semi_supervised_dbscan.pth`, `results/model_C_semi_supervised_dbscan_aug.pth`

### Étape 5 — `semi_supervised_learning_CNN.ipynb` (variante CNN avec fine-tuning)
Entraînement et comparaison des mêmes modèles A/B/C avec **ResNet50 fine-tuné** (layer4 dégelé), entraîné directement sur les images (pas de features pré-extraites).

- **Architecture** : ResNet50 avec `layer4` dégelé + tête MLP remplacée (2048 → 512 → 128 → 2).
  - `conv1`, `layer1`, `layer2`, `layer3` : **gelés** (poids ImageNet conservés, ~8,5M paramètres)
  - `layer4` : **dégelé**, entraîné avec `lr_layer4` faible (adaptation au domaine IRM)
  - Tête `fc` : **remplacée** par 3 couches Linear + BatchNorm + ReLU + Dropout
  - Total entraînable : ~16M paramètres (layer4 + tête fc)
- **HP tuning avec Optuna** : étude TPE bayésienne sur 50 trials, StratifiedKFold(k=3) sur les 80 images pré-chargées en mémoire GPU, avec MedianPruner. Espace de recherche : 5 HP simultanés.

| Hyperparamètre | Espace de recherche | Valeur retenue |
|---|---|---|
| `lr_head` | log-uniform [1e-4, 5e-3] | 2.36e-3 |
| `lr_layer4` | log-uniform [1e-6, 1e-3] | 8.20e-6 |
| `hidden_dim` | {256, 512, 1024} | 512 |
| `dropout` | uniform [0.15, 0.50] | 0.19 |
| `epochs` | int [20, 100] | 60 |

Meilleur F2-score CV (3 folds, 80 images) : **0.9749**

- **Modèle A** : supervisé pur — 60 epochs sur 80 images (labels forts).
- **Modèle B** : phase 1 (30 epochs, 1486 images DBSCAN) + phase 2 fine-tuning (30 epochs, 80 images labels forts, LR réduit).
- **Modèle C** : phase 1 (30 epochs, ~2206 images DBSCAN + augmentées) + phase 2 fine-tuning (30 epochs, 80 images labels forts, LR réduit).

- **Sorties** :
  - `data/processed/model_comparison_cnn.csv` — tableau comparatif A/B/C CNN
  - `results/confusion_matrices_cnn.png`, `results/learning_curves_cnn.png`, `results/roc_curves_cnn.png`
  - `results/model_A_cnn.pth`, `results/model_B_cnn_semi_supervised.pth`, `results/model_C_cnn_semi_supervised_aug.pth`
  - `results/optuna_hp_tuning_cnn.png` — convergence, importance des HP, lr_head vs F2

## Choix des modèles et justifications

| Modèle | Justification |
|---|---|
| **ResNet50 (backbone gelé)** | He et al. (2015) — les connexions résiduelles permettent d'entraîner des réseaux très profonds sans dégradation du gradient. Pré-entraîné sur ImageNet (1.2M images), ResNet50 produit des features transférables vers des domaines médicaux même avec peu de données labelisées (Raghu et al., 2019). |
| **ResNet50 (layer4 dégelé)** | Fine-tuning partiel : dégeler uniquement le dernier bloc résiduel (layer4) adapte les représentations de haut niveau au domaine IRM sans détruire les features générales des couches profondes. LR différencié (`lr_layer4 << lr_head`) pour préserver les poids pré-entraînés. |
| **K-Means** | Lloyd (1982) — algorithme de référence pour le clustering, rapide et adapté quand le nombre de clusters est connu (K=2 classes). Utilisé comme baseline de comparaison. |
| **DBSCAN** | Ester et al. (1996) — détecte des clusters de forme arbitraire sans fixer K à l'avance et identifie les points aberrants comme bruit. Retenu comme méthode de labels faibles (ARI=0.4200 avec eps=0.25, min_samples=3). Les points de bruit sont classés "cancer" par précaution. |
| **MLP** | Classifieur standard sur features pré-extraites (Goodfellow et al., 2016). Choisi pour sa légèreté : avec des features ResNet50 déjà discriminantes, un modèle léger suffit et réduit le risque de sur-apprentissage sur 80 images. |
| **Optuna (TPE)** | Bergstra et al. (2011) — le Tree-structured Parzen Estimator est un algorithme bayésien qui apprend des trials précédents pour concentrer la recherche dans les zones prometteuses de l'espace HP, plus efficace que la recherche par grille. |
| **Semi-supervisé** | Chapelle et al. (2006) — le semi-supervisé est particulièrement adapté quand les données labelisées sont rares, cas typique en imagerie médicale où l'annotation experte est coûteuse. Ici, 80 images labelisées vs 1406 sans label justifient cette approche. |

## Résultats (test set, 20 images)

### Variante MLP (features pré-extraites, backbone gelé)

| Modèle | Données d'entraînement | Accuracy | F2-score | ROC-AUC |
|---|---|---|---|---|
| A — Supervisé pur | 80 images (labels forts) | 0.90 | 0.8974 | 0.98 |
| B — Semi-sup. DBSCAN | 1486 images (DBSCAN weak labels) | 0.85 | 0.8440 | 0.94 |
| C — Semi-sup. DBSCAN + augmentation | ~2206 images | 0.90 | 0.8974 | 0.93 |

### Variante CNN (ResNet50 fine-tuné, layer4 dégelé)

| Modèle | Données d'entraînement | Accuracy | F2-score | ROC-AUC |
|---|---|---|---|---|
| A — Supervisé pur | 80 images (labels forts) | 0.90 | 0.8974 | 0.98 |
| B — Semi-sup. DBSCAN | 1486 images (DBSCAN weak labels) | 0.90 | 0.8974 | 0.91 |
| C — Semi-sup. DBSCAN + augmentation | ~2206 images | **0.95** | **0.9494** | **1.00** |

### Analyse comparative

- La variante CNN avec fine-tuning surpasse la variante MLP sur le modèle C (+0.05 accuracy, +0.052 F2-score).
- Le modèle C CNN est le meilleur modèle global : l'augmentation de données en phase 1 (2206 images) combinée au fine-tuning de layer4 permet d'adapter les représentations convolutives au domaine IRM.
- Les labels faibles DBSCAN seuls (modèle B) ne dégradent pas les performances en CNN (contrairement au MLP), grâce à la capacité d'adaptation de layer4.
- À noter : le test set de 20 images introduit une forte variabilité — 1 image mal classée représente 5% d'accuracy.

## Installation et exécution

Le projet utilise [uv](https://github.com/astral-sh/uv) pour la gestion des dépendances.

```bash
uv sync
uv run jupyter lab
```

Les notebooks doivent être exécutés dans l'ordre (1 → 2 → 3 → 4 → 5), chaque étape dépendant des fichiers produits par la précédente. Le notebook 5 (CNN) peut être exécuté indépendamment du notebook 4 (MLP), les deux partageant les sorties de l'étape 3.

## Configuration centralisée (`src/config.py`)

Tous les chemins (données brutes, fichiers traités, features, résultats) et hyperparamètres globaux (`RANDOM_STATE`, `N_CLUSTERS`, `DEVICE`) sont définis une seule fois dans `src/config.py` et importés par chaque notebook, afin d'éviter toute duplication ou incohérence entre les étapes.
