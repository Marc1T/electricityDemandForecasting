# ⚡️ Prévision de la Demande Électrique en Espagne

Projet académique visant à développer un système de prévision de la demande en électricité à partir de séries temporelles.  
Les données sont récupérées via l'API de Red Eléctrica de España (REE), enrichies par des sources externes (météo, jours fériés...) puis modélisées avec des approches classiques et deep learning.

---
```
energy_demand_forecasting/
│
├── 📁 data/
│   ├── 📁 raw/                     # Données brutes collectées depuis l’API REE
│   ├── 📁 interim/                 # Données prétraitées : fusionnées, nettoyées, enrichies
│   ├── 📁 processed/               # Données finales prêtes pour l'entraînement
│   ├── 📁 external/                # Données exogènes : météo, jours fériés, prix énergie, etc.
│   └── 📁 submission/              # Prédictions à soumettre (le cas échéant)
│
├── 📁 notebooks/
│   ├── 01_data_retrieval.ipynb    # Récupération API REE + collecte sources externes
│   ├── 02_preprocessing.ipynb     # Nettoyage, mise en forme, création features temporelles
│   ├── 03_feature_engineering.ipynb # Création des variables explicatives
│   ├── 04_eda.ipynb               # Analyse exploratoire, corrélations, visualisations
│   ├── 05_modeling_baseline.ipynb # Modèles classiques : régression linéaire, RandomForest, etc.
│   ├── 06_modeling_dl.ipynb       # Deep Learning : LSTM, GRU, TCN, Transformer
│   ├── 07_evaluation.ipynb        # Comparaison modèles, métriques, erreurs, robustesse
│   └── 08_dashboard.ipynb         # Application de visualisation interactive (Streamlit / Dash)
│
├── 📁 src/                        # Code source modulaire en Python
│   ├── data/
│   │   ├── data_loading.py        # Téléchargement API REE, chargement CSV, fusion
│   │   ├── external_data.py       # Collecte données météo, calendaires, autres API
│   │   └── preprocessing.py       # Nettoyage, formatage des dates, ré-échantillonnage
│   ├── features/
│   │   └── feature_engineering.py # Création des variables explicatives (lag, rolling, etc.)
│   ├── models/
│   │   ├── modeling.py            # Entraînement modèles ML et DL
│   │   ├── tuning.py              # Optimisation d’hyperparamètres
│   │   └── evaluation.py          # RMSE, MAE, MAPE, courbes, etc.
│   └── utils/
│       └── utils.py               # Fonctions générales : logs, timer, visualisation commune
│
├── 📁 models/                     # Sauvegarde des modèles entraînés (pickle, joblib, .h5)
│
├── 📁 outputs/
│   ├── 📁 figures/                # Graphiques d’analyse, courbes de prévision
│   └── 📁 reports/                # Résultats des expériences, résumé des performances
│
├── 📁 dashboard/                  # Code de l'app Streamlit/Dash : visualisation dynamique
│   ├── app.py                    # Script principal
│   └── components.py             # Composants personnalisés (graphiques, widgets)
│
├── requirements.txt              # Dépendances Python (pandas, sklearn, keras, streamlit…)
├── README.md                     # Présentation du projet, installation, usage
└── config.yaml                   # Paramètres globaux : chemins, variables cibles, configs API, etc.
```
---

## 🔧 Étapes principales

1. **Collecte de données** via l’API REE et autres sources
2. **Nettoyage & feature engineering** sur les séries temporelles
3. **Analyse exploratoire** (EDA)
4. **Modélisation** : Régression, Random Forest, LSTM, TCN, etc.
5. **Évaluation** des performances (MAPE, RMSE…)
6. **Déploiement** d’un dashboard interactif

---

## 📊 Objectif

Prédire la demande électrique horaire ou journalière, pour appuyer la planification énergétique et l’optimisation des ressources.

---

## 🧑‍💻 Auteur

- **Marc Thierry NANKOULI**
- **Abdoulaye**
- **Halima Elfilali Ech-Chafiq**
- Étudiants en IA & Data Technologies
- Contact : consultasios@ree.es (pour obtenir la clé API fournie par REE)

