import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# =========================================================
# 1. PARAMÈTRES
# =========================================================

CSV_PATH = "DA286_1990_2025_returns_Monthly_with_RF.csv"

PORTFOLIO_SIZES = [2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50,
                   60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
                   180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 286]

N_PORTFOLIOS_PER_SIZE = 10000
print("Nombre de portefeuilles généré par niveau de concentration:", N_PORTFOLIOS_PER_SIZE)
RANDOM_SEED = 42

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"outputs_strategy1_rf_{timestamp}")
OUTPUT_DIR.mkdir(exist_ok=True)


# 2. CHARGEMENT DES DONNÉES
# =========================================================

df = pd.read_csv(CSV_PATH)
df["Date"] = df["Date"].astype(str)

# Séparer les actions et le RF
asset_cols = [c for c in df.columns if c not in ["Date", "RF"]]
returns_df = df[asset_cols].astype(float)
rf_series = df["RF"].astype(float).to_numpy()

n_months, n_assets = returns_df.shape

PORTFOLIO_SIZES = [k for k in PORTFOLIO_SIZES if k <= n_assets]

print(f"Nombre de mois : {n_months}")
print(f"Nombre d'actions : {n_assets}")
print(f"Période : {df['Date'].iloc[0]} à {df['Date'].iloc[-1]}")
print(f"Niveaux de concentration retenus : {PORTFOLIO_SIZES}")

# Vérification RF
print("\nAperçu RF :")
print(df[["Date", "RF"]].head())
print(f"Valeurs manquantes RF : {df['RF'].isna().sum()}")

# =========================================================
# 3. FONCTIONS UTILES
# =========================================================

def annualized_return_from_monthly(portfolio_returns: np.ndarray) -> float:
    growth = np.prod(1 + portfolio_returns)
    n = len(portfolio_returns)
    return growth ** (12 / n) - 1


def annualized_volatility_from_monthly(portfolio_returns: np.ndarray) -> float:
    return np.std(portfolio_returns, ddof=1) * np.sqrt(12)


def annualized_excess_return_from_monthly(portfolio_returns: np.ndarray, rf_array: np.ndarray) -> float:
    """
    Rendement excédentaire annualisé à partir de rendements mensuels.
    """
    excess = portfolio_returns - rf_array
    growth = np.prod(1 + excess)
    n = len(excess)
    return growth ** (12 / n) - 1


def sharpe_ratio(portfolio_returns: np.ndarray, rf_array: np.ndarray) -> float:
    """
    Ratio de Sharpe annualisé avec RF mensuel variable.
    """
    vol_ann = annualized_volatility_from_monthly(portfolio_returns)
    if np.isclose(vol_ann, 0):
        return np.nan
    excess_ann = annualized_excess_return_from_monthly(portfolio_returns, rf_array)
    return excess_ann / vol_ann


def compute_buy_and_hold_returns(asset_return_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule les rendements mensuels buy-and-hold d'un portefeuille équipondéré au départ.
    Retourne :
    - portfolio_returns : array (T,)
    - weights_over_time : array (T, N), poids au début de chaque mois
    """
    T, N = asset_return_matrix.shape
    weights = np.ones(N) / N

    portfolio_returns = np.zeros(T)
    weights_over_time = np.zeros((T, N))

    for t in range(T):
        weights_over_time[t] = weights.copy()

        # rendement du portefeuille au mois t
        r_t = np.dot(weights, asset_return_matrix[t])
        portfolio_returns[t] = r_t

        # mise à jour des poids buy-and-hold pour le mois suivant
        gross_returns = 1 + asset_return_matrix[t]
        new_values = weights * gross_returns
        total_value = new_values.sum()

        if total_value <= 0:
            weights = np.ones(N) / N
        else:
            weights = new_values / total_value

    return portfolio_returns, weights_over_time


def hhi(weights: np.ndarray) -> float:
    return np.sum(weights ** 2)


def enp(weights: np.ndarray) -> float:
    h = hhi(weights)
    return np.nan if np.isclose(h, 0) else 1 / h


def simulate_one_portfolio(returns_df: pd.DataFrame,
                           rf_array: np.ndarray,
                           k: int,
                           rng: np.random.Generator) -> dict:
    """
    Simule un portefeuille de k actions tirées aléatoirement.
    """
    chosen_assets = rng.choice(returns_df.columns.to_numpy(), size=k, replace=False)
    sub = returns_df[chosen_assets].to_numpy()

    # Rendements buy-and-hold
    p_returns, weights_t = compute_buy_and_hold_returns(sub)

    # Concentration initiale
    w0 = np.ones(k) / k
    hhi_init = hhi(w0)
    enp_init = enp(w0)

    # Concentration moyenne buy-and-hold
    hhi_mean = np.mean([hhi(w) for w in weights_t])
    enp_mean = np.mean([enp(w) for w in weights_t])

    # Performance
    ann_return = annualized_return_from_monthly(p_returns)
    ann_vol = annualized_volatility_from_monthly(p_returns)
    sr = sharpe_ratio(p_returns, rf_array)

    return {
        "n_assets": k,
        "assets": list(chosen_assets),
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sr,
        "HHI_init": hhi_init,
        "ENP_init": enp_init,
        "HHI_mean_BH": hhi_mean,
        "ENP_mean_BH": enp_mean,
    }


# =========================================================
# 4. SIMULATION PRINCIPALE
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)

all_results = []

for k in PORTFOLIO_SIZES:
    print(f"Simulation des portefeuilles avec {k} actions...")
    for i in range(N_PORTFOLIOS_PER_SIZE):
        res = simulate_one_portfolio(
            returns_df=returns_df,
            rf_array=rf_series,
            k=k,
            rng=rng
        )
        res["portfolio_id_within_group"] = i + 1
        all_results.append(res)

results_df = pd.DataFrame(all_results)

# Sauvegarde détaillée
results_df.to_csv(OUTPUT_DIR / "portfolio_level_results.csv", index=False)

# =========================================================
# 5. AGRÉGATION PAR NIVEAU DE CONCENTRATION
# =========================================================

summary_df = (
    results_df
    .groupby("n_assets")
    .agg(
        n_portfolios=("n_assets", "size"),

        mean_return=("annualized_return", "mean"),
        median_return=("annualized_return", "median"),
        std_return=("annualized_return", "std"),

        mean_volatility=("annualized_volatility", "mean"),
        median_volatility=("annualized_volatility", "median"),
        std_volatility=("annualized_volatility", "std"),

        mean_sharpe=("sharpe_ratio", "mean"),
        median_sharpe=("sharpe_ratio", "median"),
        std_sharpe=("sharpe_ratio", "std"),

        mean_HHI_init=("HHI_init", "mean"),
        mean_ENP_init=("ENP_init", "mean"),

        mean_HHI_BH=("HHI_mean_BH", "mean"),
        mean_ENP_BH=("ENP_mean_BH", "mean"),
    )
    .reset_index()
)

summary_df.to_csv(OUTPUT_DIR / "summary_by_concentration.csv", index=False)

print("\nRésumé agrégé :")
print(summary_df.head())

# =========================================================
# 6. GRAPHIQUES
# =========================================================

def save_line_plot(x, y, ylabel, title, filename, y_std=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    if y_std is not None:
        y = np.asarray(y)
        y_std = np.asarray(y_std)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)
    plt.xlabel("Nombre d'actions dans le portefeuille")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

save_line_plot(
    x=summary_df["n_assets"],
    y=summary_df["mean_return"],
    y_std=summary_df["std_return"],
    ylabel="Rendement annualisé moyen",
    title="Évolution du rendement annualisé selon la concentration",
    filename="mean_return_vs_concentration.png"
)

save_line_plot(
    x=summary_df["n_assets"],
    y=summary_df["mean_volatility"],
    y_std=summary_df["std_volatility"],
    ylabel="Volatilité annualisée moyenne",
    title="Évolution de la volatilité selon la concentration",
    filename="mean_volatility_vs_concentration.png"
)

save_line_plot(
    x=summary_df["n_assets"],
    y=summary_df["mean_sharpe"],
    y_std=summary_df["std_sharpe"],
    ylabel="Ratio de Sharpe moyen",
    title="Évolution du ratio de Sharpe selon la concentration",
    filename="mean_sharpe_vs_concentration.png"
)

save_line_plot(
    x=summary_df["n_assets"],
    y=summary_df["mean_HHI_BH"],
    ylabel="HHI moyen (buy and hold)",
    title="Évolution du HHI moyen selon la concentration initiale",
    filename="mean_hhi_bh_vs_concentration.png"
)

save_line_plot(
    x=summary_df["n_assets"],
    y=summary_df["mean_ENP_BH"],
    ylabel="ENP moyen (buy and hold)",
    title="Évolution de l'ENP moyen selon la concentration initiale",
    filename="mean_enp_bh_vs_concentration.png"
)

print(f"\nTous les résultats ont été sauvegardés dans : {OUTPUT_DIR.resolve()}")