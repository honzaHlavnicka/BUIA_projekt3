"""Pomocné funkce pro načítání dat, rozdělení datasetu a výpočet metrik."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Union, List


def load_data(filepath: str) -> pd.DataFrame:
    """Načte dataset ze souboru CSV."""
    df = pd.read_csv(filepath)
    return df


def train_test_split_9_1(
    df: pd.DataFrame,
    target_col: str = "cena",
    feature_col: Union[str, List[str]] = "plocha",
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Rozdělí dataset na trénovací a testovací data v zadaném poměru (výchozí 9:1).
    
    Vrací:
        X_train, y_train, X_test, y_test, df_train, df_test
    """
    # Fixace náhodného generátoru pro reprodukovatelnost
    rng = np.random.default_rng(random_state)
    n = len(df)
    test_size = int(np.round(n * test_ratio))
    
    indices = np.arange(n)
    rng.shuffle(indices)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    
    X_train = df_train[feature_col].to_numpy(dtype=np.float64)
    y_train = df_train[target_col].to_numpy(dtype=np.float64)
    X_test = df_test[feature_col].to_numpy(dtype=np.float64)
    y_test = df_test[target_col].to_numpy(dtype=np.float64)
    
    return X_train, y_train, X_test, y_test, df_train, df_test


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypočítá střední kvadratickou chybu (Mean Squared Error - MSE)."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypočítá průměrnou absolutní chybu (Mean Absolute Error - MAE)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypočítá koeficient determinace (R² skóre)."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Vrátí slovník se základními metrikami (MSE, MAE, RMSE, R2)."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
