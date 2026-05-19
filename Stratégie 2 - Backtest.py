import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# =========================================================
# 1. PARAMÈTRES
# =========================================================

CSV_PATH = "DA286_1990_2025_returns_Monthly_with_RF.csv"

PORTFOLIO_SIZES = [2, 3, 4, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50,
                   60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
                   180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 286]

LOOKBACK_MONTHS = 36
CANDIDATES_PER_K = 200
print("Nombre de candidats par niveau de concentration:", CANDIDATES_PER_K)
RANDOM_SEED = 10

# Critère de sélection in-sample :
# "sharpe", "return", "min_vol"
SELECTION_METRIC = "sharpe"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"outputs_backtest_rf_{timestamp}")
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
PORTFOLIO_SIZES = [k for k in PORTFOLIO_SIZES if k <= N]

print(f"Période : {df['Date'].iloc[0]} à {df['Date'].iloc[-1]}")
print(f"Nombre de mois : {T}")
print(f"Nombre d'actions : {N}")
print(f"Niveaux de concentration : {PORTFOLIO_SIZES}")
print(f"Valeurs manquantes RF : {df['RF'].isna().sum()}")

# =========================================================
# 3. FONCTIONS UTILES
# =========================================================

def annualized_return_from_monthly(r: np.ndarray) -> float:
    if len(r) == 0:
        return np.nan
    growth = np.prod(1 + r)
    n = len(r)
    return growth ** (12 / n) - 1


def annualized_volatility_from_monthly(r: np.ndarray) -> float:
    if len(r) < 2:
        return np.nan
    return np.std(r, ddof=1) * np.sqrt(12)


def annualized_excess_return_from_monthly(r: np.ndarray, rf: np.ndarray) -> float:
    if len(r) == 0:
        return np.nan
    excess = r - rf
    growth = np.prod(1 + excess)
    n = len(excess)
    return growth ** (12 / n) - 1


def annualized_sharpe_from_monthly(r: np.ndarray, rf: np.ndarray) -> float:
    if len(r) < 2:
        return np.nan
    vol = annualized_volatility_from_monthly(r)
    if np.isnan(vol) or np.isclose(vol, 0):
        return np.nan
    excess_ann = annualized_excess_return_from_monthly(r, rf)
    return excess_ann / vol


def score_candidate(candidate_returns: np.ndarray, metric: str, rf_window: np.ndarray) -> float:
    if metric == "sharpe":
        return annualized_sharpe_from_monthly(candidate_returns, rf_window)
    elif metric == "return":
        return annualized_return_from_monthly(candidate_returns)
    elif metric == "min_vol":
        vol = annualized_volatility_from_monthly(candidate_returns)
        return -vol if not np.isnan(vol) else np.nan
    else:
        raise ValueError("SELECTION_METRIC doit être 'sharpe', 'return' ou 'min_vol'")


def hhi(weights: np.ndarray) -> float:
    return np.sum(weights ** 2)


def enp(weights: np.ndarray) -> float:
    h = hhi(weights)
    return np.nan if np.isclose(h, 0) else 1 / h


def equal_weight_portfolio_returns(window_df: pd.DataFrame, selected_assets: list[str]) -> np.ndarray:
    """
    Rendements mensuels d'un portefeuille équipondéré rebalancé implicitement à chaque mois.
    Pour un scoring in-sample simple, c'est propre et défendable.
    """
    return window_df[selected_assets].mean(axis=1).to_numpy()


# =========================================================
# 4. BACKTEST ROLLING
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)
oos_rows = []

# t = mois testé ; fenêtre d'estimation = [t-LOOKBACK_MONTHS, t-1]
for t in range(LOOKBACK_MONTHS, T):
    train_df = returns_df.iloc[t - LOOKBACK_MONTHS:t]
    train_rf = rf_series[t - LOOKBACK_MONTHS:t]

    test_row = returns_df.iloc[t]
    test_rf = rf_series[t]
    test_date = df["Date"].iloc[t]

    print(f"Backtest mois test : {test_date}")

    for k in PORTFOLIO_SIZES:

        valid_assets = train_df.columns[
            train_df.notna().all(axis=0) & test_row.notna()
        ].tolist()

        if len(valid_assets) < k:
            continue

        best_score = -np.inf
        best_assets = None

        for _ in range(CANDIDATES_PER_K):
            selected = rng.choice(valid_assets, size=k, replace=False).tolist()

            candidate_train_returns = equal_weight_portfolio_returns(train_df, selected)
            score = score_candidate(candidate_train_returns, SELECTION_METRIC, train_rf)

            if np.isnan(score):
                continue

            if score > best_score:
                best_score = score
                best_assets = selected

        if best_assets is None:
            continue

        # Out-of-sample : rendement du mois suivant
        oos_return = test_row[best_assets].mean()
        oos_excess_return = oos_return - test_rf

        w0 = np.ones(k) / k

        oos_rows.append({
            "Date": test_date,
            "n_assets": k,
            "selection_metric": SELECTION_METRIC,
            "lookback_months": LOOKBACK_MONTHS,
            "candidates_per_k": CANDIDATES_PER_K,
            "best_in_sample_score": best_score,
            "oos_return": oos_return,
            "oos_excess_return": oos_excess_return,
            "RF_test_month": test_rf,
            "HHI_init": hhi(w0),
            "ENP_init": enp(w0),
            "selected_assets": "|".join(best_assets)
        })

oos_df = pd.DataFrame(oos_rows)
oos_df.to_csv(OUTPUT_DIR / "oos_monthly_results.csv", index=False)

# =========================================================
# 5. RÉSUMÉ PAR NIVEAU DE CONCENTRATION
# =========================================================

summary_rows = []

for k in sorted(oos_df["n_assets"].unique()):
    sub = oos_df[oos_df["n_assets"] == k].copy()

    r = sub["oos_return"].to_numpy()
    rf = sub["RF_test_month"].to_numpy()
    excess_r = sub["oos_excess_return"].to_numpy()

    summary_rows.append({
        "n_assets": k,
        "n_obs": len(sub),
        "annualized_return_oos": annualized_return_from_monthly(r),
        "annualized_excess_return_oos": annualized_return_from_monthly(excess_r),
        "annualized_volatility_oos": annualized_volatility_from_monthly(r),
        "sharpe_oos": annualized_sharpe_from_monthly(r, rf),
        "mean_monthly_return_oos": np.mean(r),
        "mean_monthly_excess_return_oos": np.mean(excess_r),
        "std_monthly_return_oos": np.std(r, ddof=1) if len(r) > 1 else np.nan,
        "HHI_init": sub["HHI_init"].iloc[0],
        "ENP_init": sub["ENP_init"].iloc[0]
    })

summary_df = pd.DataFrame(summary_rows).sort_values("n_assets")
summary_df.to_csv(OUTPUT_DIR / "summary_oos_by_concentration.csv", index=False)

print("\nRésumé agrégé :")
print(summary_df.head())

# =========================================================
# 6. GRAPHIQUES PRINCIPAUX
# =========================================================

def save_line_plot(x, y, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o")
    plt.xlabel("Nombre d'actions dans le portefeuille")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


save_line_plot(
    summary_df["n_assets"],
    summary_df["annualized_return_oos"],
    "Rendement annualisé out-of-sample",
    "Backtest : rendement annualisé selon la concentration",
    "backtest_return_oos.png"
)

save_line_plot(
    summary_df["n_assets"],
    summary_df["annualized_excess_return_oos"],
    "Rendement excédentaire annualisé out-of-sample",
    "Backtest : rendement excédentaire selon la concentration",
    "backtest_excess_return_oos.png"
)

save_line_plot(
    summary_df["n_assets"],
    summary_df["annualized_volatility_oos"],
    "Volatilité annualisée out-of-sample",
    "Backtest : volatilité selon la concentration",
    "backtest_volatility_oos.png"
)

save_line_plot(
    summary_df["n_assets"],
    summary_df["sharpe_oos"],
    "Ratio de Sharpe out-of-sample",
    "Backtest : ratio de Sharpe selon la concentration",
    "backtest_sharpe_oos.png"
)

# =========================================================
# 7. COURBES DE PERFORMANCE CUMULÉE
# =========================================================

pivot_oos = oos_df.pivot(index="Date", columns="n_assets", values="oos_return").sort_index()

# Conversion des dates pour rendre l'axe horizontal plus lisible
pivot_oos.index = pd.to_datetime(pivot_oos.index.astype(str), format="%Y%m")

cum_perf = (1 + pivot_oos).cumprod()

plt.figure(figsize=(12, 7))
for k in summary_df["n_assets"]:
    plt.plot(cum_perf.index, cum_perf[k], alpha=0.5)

plt.title("Backtest : performance cumulée out-of-sample par niveau de concentration")
plt.xlabel("Année")
plt.ylabel("Valeur d’un investissement initial (base 1)")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_cumulative_performance_all.png", dpi=300)
plt.close()

selected_levels_for_plot = [2, 5, 10, 25, 50, 100, 150, 200, 250,286]
selected_levels_for_plot = [k for k in selected_levels_for_plot if k in cum_perf.columns]

plt.figure(figsize=(12, 7))
for k in selected_levels_for_plot:
    plt.plot(cum_perf.index, cum_perf[k], label=f"{k} actions")

plt.title("Backtest : performance cumulée out-of-sample (niveaux sélectionnés)")
plt.xlabel("Année")
plt.ylabel("Valeur d’un investissement initial (base 1)")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "backtest_cumulative_performance_selected.png", dpi=300)
plt.close()

print(f"\nRésultats sauvegardés dans : {OUTPUT_DIR.resolve()}")