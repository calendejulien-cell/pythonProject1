import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

# =========================================================
# 1. PARAMÈTRES
# =========================================================

CSV_PATH = "DA286_1990_2025_returns_Monthly_with_RF.csv"

LOOKBACK_MONTHS = 36
RANDOM_SEED = 42

# Pour limiter la taille du problème d'optimisation :
# on peut optimiser sur un sous-univers fixe d'actions
# ou sur tout l'univers
USE_SUBSET = True
SUBSET_SIZE = 286   # ex: 50, 100, 150 ; si False, utilise tout l'univers

# Optionnel : limiter le poids max par action pour éviter des solutions extrêmes
USE_MAX_WEIGHT_CONSTRAINT = True
MAX_WEIGHT = 0.10  # ex: max 10% par action

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"outputs_markowitz_{timestamp}")
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# 2. CHARGEMENT DES DONNÉES
# =========================================================

df = pd.read_csv(CSV_PATH)
df["Date"] = df["Date"].astype(str)

asset_cols = [c for c in df.columns if c not in ["Date", "RF"]]
returns_df = df[asset_cols].astype(float)
rf_series = df["RF"].astype(float).to_numpy()

T, N = returns_df.shape

print(f"Période : {df['Date'].iloc[0]} à {df['Date'].iloc[-1]}")
print(f"Nombre de mois : {T}")
print(f"Nombre d'actions disponibles : {N}")
print(f"Valeurs manquantes RF : {df['RF'].isna().sum()}")

# =========================================================
# 3. FONCTIONS UTILES
# =========================================================

def annualized_return_from_monthly(r: np.ndarray) -> float:
    if len(r) == 0:
        return np.nan
    growth = np.prod(1 + r)
    return growth ** (12 / len(r)) - 1


def annualized_volatility_from_monthly(r: np.ndarray) -> float:
    if len(r) < 2:
        return np.nan
    return np.std(r, ddof=1) * np.sqrt(12)


def annualized_excess_return_from_monthly(r: np.ndarray, rf: np.ndarray) -> float:
    if len(r) == 0:
        return np.nan
    excess = r - rf
    growth = np.prod(1 + excess)
    return growth ** (12 / len(excess)) - 1


def annualized_sharpe_from_monthly(r: np.ndarray, rf: np.ndarray) -> float:
    vol = annualized_volatility_from_monthly(r)
    if np.isnan(vol) or np.isclose(vol, 0):
        return np.nan
    excess_ann = annualized_excess_return_from_monthly(r, rf)
    return excess_ann / vol


def hhi(weights: np.ndarray) -> float:
    return np.sum(weights ** 2)


def enp(weights: np.ndarray) -> float:
    hh = hhi(weights)
    return np.nan if np.isclose(hh, 0) else 1 / hh


def markowitz_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, max_weight=None):
    """
    Optimisation Markowitz long-only :
    maximise le Sharpe ex-ante sous contraintes:
    - somme des poids = 1
    - poids >= 0
    - éventuellement poids <= max_weight
    """

    n = len(mu)

    # Petite régularisation diagonale pour stabilité numérique
    cov_reg = cov + np.eye(n) * 1e-8

    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov_reg, w)))
        if np.isclose(port_vol, 0):
            return 1e6
        return - (port_return - rf) / port_vol

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    if max_weight is None:
        bounds = [(0.0, 1.0) for _ in range(n)]
    else:
        bounds = [(0.0, max_weight) for _ in range(n)]

    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return None

    return result.x


# =========================================================
# 4. CHOIX DU SOUS-UNIVERS
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

if USE_SUBSET:
    chosen_assets = rng.choice(asset_cols, size=min(SUBSET_SIZE, len(asset_cols)), replace=False).tolist()
    working_returns_df = returns_df[chosen_assets].copy()
    print(f"Sous-univers utilisé : {len(chosen_assets)} actions")
else:
    chosen_assets = asset_cols
    working_returns_df = returns_df.copy()
    print(f"Univers complet utilisé : {len(chosen_assets)} actions")

# =========================================================
# 5. BACKTEST MARKOWITZ ROLLING
# =========================================================

results = []

for t in range(LOOKBACK_MONTHS, T):
    train_df = working_returns_df.iloc[t - LOOKBACK_MONTHS:t].copy()
    test_row = working_returns_df.iloc[t].copy()

    train_rf = rf_series[t - LOOKBACK_MONTHS:t]
    test_rf = rf_series[t]

    test_date = df["Date"].iloc[t]

    # On ne garde que les actions sans NA sur la fenêtre et sur le mois test
    valid_assets = train_df.columns[
        train_df.notna().all(axis=0) & test_row.notna()
    ].tolist()

    if len(valid_assets) < 2:
        continue

    train_matrix = train_df[valid_assets].to_numpy()
    test_vector = test_row[valid_assets].to_numpy()

    # Estimation moments
    mu = train_matrix.mean(axis=0)              # rendements moyens mensuels
    cov = np.cov(train_matrix, rowvar=False)    # covariance mensuelle
    rf_in_sample = np.mean(train_rf)            # RF mensuel moyen sur la fenêtre

    # Optimisation
    weights = markowitz_max_sharpe(
        mu=mu,
        cov=cov,
        rf=rf_in_sample,
        max_weight=MAX_WEIGHT if USE_MAX_WEIGHT_CONSTRAINT else None
    )

    if weights is None:
        continue

    # Rendement ex-post du mois suivant
    oos_return = np.dot(weights, test_vector)
    oos_excess_return = oos_return - test_rf

    # Statistiques ex-ante du portefeuille optimisé
    ex_ante_return = np.dot(weights, mu)
    ex_ante_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    ex_ante_sharpe = np.nan
    if not np.isclose(ex_ante_vol, 0):
        ex_ante_sharpe = (ex_ante_return - rf_in_sample) / ex_ante_vol

    # Concentration
    portfolio_hhi = hhi(weights)
    portfolio_enp = enp(weights)
    n_nonzero = np.sum(weights > 1e-8)

    results.append({
        "Date": test_date,
        "n_assets_available": len(valid_assets),
        "n_assets_nonzero": int(n_nonzero),
        "RF_test_month": test_rf,
        "oos_return": oos_return,
        "oos_excess_return": oos_excess_return,
        "ex_ante_return": ex_ante_return,
        "ex_ante_vol": ex_ante_vol,
        "ex_ante_sharpe": ex_ante_sharpe,
        "HHI": portfolio_hhi,
        "ENP": portfolio_enp,
        "max_weight": np.max(weights),
        "weights": "|".join([f"{a}:{w:.6f}" for a, w in zip(valid_assets, weights) if w > 1e-8])
    })

markowitz_df = pd.DataFrame(results)
markowitz_df.to_csv(OUTPUT_DIR / "markowitz_monthly_results.csv", index=False)

print("\nAperçu des résultats mensuels :")
print(markowitz_df.head())

# =========================================================
# 6. STATISTIQUES GLOBALES
# =========================================================

r = markowitz_df["oos_return"].to_numpy()
rf = markowitz_df["RF_test_month"].to_numpy()
excess_r = markowitz_df["oos_excess_return"].to_numpy()

summary = {
    "lookback_months": LOOKBACK_MONTHS,
    "subset_used": USE_SUBSET,
    "subset_size": SUBSET_SIZE if USE_SUBSET else len(chosen_assets),
    "max_weight_constraint": USE_MAX_WEIGHT_CONSTRAINT,
    "max_weight": MAX_WEIGHT if USE_MAX_WEIGHT_CONSTRAINT else np.nan,
    "n_oos_months": len(markowitz_df),
    "annualized_return_oos": annualized_return_from_monthly(r),
    "annualized_excess_return_oos": annualized_return_from_monthly(excess_r),
    "annualized_volatility_oos": annualized_volatility_from_monthly(r),
    "sharpe_oos": annualized_sharpe_from_monthly(r, rf),
    "mean_HHI": markowitz_df["HHI"].mean(),
    "mean_ENP": markowitz_df["ENP"].mean(),
    "mean_nonzero_assets": markowitz_df["n_assets_nonzero"].mean(),
    "mean_max_weight": markowitz_df["max_weight"].mean()
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT_DIR / "markowitz_summary.csv", index=False)

print("\nRésumé global Markowitz :")
print(summary_df.T)

# =========================================================
# 7. GRAPHIQUES
# =========================================================

# Performance cumulée
markowitz_df["cum_perf"] = (1 + markowitz_df["oos_return"]).cumprod()
markowitz_df["cum_excess_perf"] = (1 + markowitz_df["oos_excess_return"]).cumprod()

plt.figure(figsize=(11, 6))
plt.plot(markowitz_df["Date"], markowitz_df["cum_perf"])
plt.title("Markowitz : performance cumulée out-of-sample")
plt.xlabel("Date")
plt.ylabel("Valeur cumulée")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "markowitz_cumulative_performance.png", dpi=300)
plt.close()

# Concentration dynamique
plt.figure(figsize=(11, 6))
plt.plot(markowitz_df["Date"], markowitz_df["HHI"], label="HHI")
plt.plot(markowitz_df["Date"], markowitz_df["ENP"], label="ENP")
plt.title("Markowitz : évolution de la concentration")
plt.xlabel("Date")
plt.ylabel("Niveau")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "markowitz_concentration_over_time.png", dpi=300)
plt.close()

# Nombre effectif d'actifs
plt.figure(figsize=(11, 6))
plt.plot(markowitz_df["Date"], markowitz_df["n_assets_nonzero"])
plt.title("Markowitz : nombre d'actifs effectivement détenus")
plt.xlabel("Date")
plt.ylabel("Nombre d'actifs non nuls")
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "markowitz_nonzero_assets.png", dpi=300)
plt.close()

print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR.resolve()}")