# Analyse de Portefeuilles - Diversification & Concentration

Ce projet analyse l'impact de la concentration (nombre d'actions) sur la performance de portefeuilles actions, à travers trois stratégies distinctes. Les données couvrent 286 actions sur la période 1990-2025 (rendements mensuels).

## Données requises

Placer le fichier `DA286_1990_2025_returns_Monthly_with_RF.csv` à la racine du projet.
Ce fichier doit contenir une colonne `Date`, une colonne `RF` (taux sans risque mensuel), et une colonne par action.

## Installation des dépendances

```bash
pip install pandas numpy matplotlib scipy
```

## Lancer les scripts

Chaque script est indépendant et peut être exécuté directement :

```bash
python "Stratégie 1 - Equipondérée.py"
python "Stratégie 2 - Backtest.py"
python "Stratégie 3 - Markowitz.py"
```

Les résultats (CSV + graphiques PNG) sont sauvegardés automatiquement dans un dossier horodaté créé à la racine.

---

## Contenu des fichiers

### `Stratégie 1 - Equipondérée.py`
Simule **10 000 portefeuilles aléatoires par niveau de concentration** (de 2 à 286 actions), avec une pondération équipondérée en buy-and-hold (pas de rééquilibrage).
Calcule pour chaque portefeuille : rendement annualisé, volatilité, ratio de Sharpe, HHI et ENP.
Produit un résumé agrégé par niveau de concentration et 5 graphiques illustrant l'évolution des métriques selon le nombre d'actions.

### `Stratégie 2 - Backtest.py`
Effectue un **backtest rolling out-of-sample** : chaque mois, sélectionne parmi 200 candidats aléatoires le portefeuille équipondéré ayant le meilleur score in-sample (Sharpe par défaut) sur les 36 mois précédents, puis mesure son rendement le mois suivant.
Couvre tous les niveaux de concentration testés en parallèle. Produit des courbes de performance cumulée et un résumé OOS par niveau de concentration.

### `Stratégie 3 - Markowitz.py`
Applique une **optimisation Markowitz (maximisation du Sharpe ex-ante)** en rolling sur 36 mois, sur un sous-univers de 286 actions.
Les poids sont optimisés via SLSQP (long-only, contrainte de poids max à 10% par action). Mesure la performance OOS mensuelle, la concentration dynamique (HHI/ENP) et le nombre d'actifs effectivement détenus.
