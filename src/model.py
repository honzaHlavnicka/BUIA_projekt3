"""Implementace jednoduchého lineárního regresního modelu f(x) = ax + b
trénovaného pomocí gradientního sestupu (Gradient Descent).
"""

import json
import pickle
import time
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


class LinearRegressionGD:
    """
    Jednoduchý lineární regresní model f(x) = a * x + b.
    
    Parametry:
        learning_rate (float): Rychlost učení (krok gradientního sestupu).
        epochs (int): Celkový počet trénovacích epoch (iterací).
        log_interval (int): Po kolika epochách se vypisuje stav do terminálu.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1000,
        log_interval: int = 100,
        feature_name: str = "plocha",
        target_name: str = "cena",
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log_interval = log_interval
        self.feature_name = feature_name
        self.target_name = target_name
        
        # Váhy modelu: f(x) = a * x + b
        self.a: float = 0.0
        self.b: float = 0.0
        
        # Škálovací parametry (normalizace vstupů pro numerickou stabilitu GD)
        self.x_mean: float = 0.0
        self.x_std: float = 1.0
        
        # Vnitřní váhy na škálovaných datech z = (x - mean) / std
        self.w_z: float = 0.0
        self.b_z: float = 0.0
        
        # Referenční analytické řešení (OLS) pro kontrolu optimality
        self.ols_a: Optional[float] = None
        self.ols_b: Optional[float] = None
        
        # Historie trénování
        self.history: List[Dict[str, Any]] = []

    def _update_unscaled_weights(self) -> None:
        """Přepočítá vnitřní váhy zpět do původní rovnice f(x) = a * x + b."""
        self.a = float(self.w_z / self.x_std)
        self.b = float(self.b_z - (self.w_z * self.x_mean / self.x_std))

    def _compute_ols(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Vypočítá analytické optimální řešení metodou nejmenších čtverců (OLS)."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.sum((x - x_mean) ** 2)
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        
        a_ols = float(cov_xy / var_x) if var_x != 0 else 0.0
        b_ols = float(y_mean - a_ols * x_mean)
        return a_ols, b_ols

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        step_delay: float = 0.0,
        verbose: bool = True,
    ) -> "LinearRegressionGD":
        """
        Natrénuje lineární model f(x) = a * x + b pomocí gradientního sestupu.
        Průběžně vypisuje stav trénování a přesnost.
        """
        x_tr = np.asarray(X_train, dtype=np.float64).flatten()
        y_tr = np.asarray(y_train, dtype=np.float64).flatten()
        n = len(x_tr)
        
        # Analytické OLS řešení pro porovnání
        self.ols_a, self.ols_b = self._compute_ols(x_tr, y_tr)
        
        # Výpočet průměru a směrodatné odchylky pro standardizaci
        self.x_mean = float(np.mean(x_tr))
        self.x_std = float(np.std(x_tr))
        if self.x_std == 0:
            self.x_std = 1.0
            
        z_tr = (x_tr - self.x_mean) / self.x_std
        
        # Inicializace vah
        self.w_z = 0.0
        self.b_z = float(np.mean(y_tr))
        self._update_unscaled_weights()
        self.history = []
        
        if verbose:
            print("=" * 105)
            print(" Zahájení trénování jednoduchého lineárního modelu f(x) = a * x + b (Gradient Descent)")
            print(f" Dataset: {n} trénovacích vzorků" + (f", {len(X_test)} testovacích vzorků" if X_test is not None else ""))
            print(f" Hyperparametry: learning_rate={self.learning_rate}, epochs={self.epochs}, log_interval={self.log_interval}")
            print(f" Analytické OLS optimum: f(x) = {self.ols_a:.6f} * x + {self.ols_b:.6f}")
            print("=" * 105)
            header = f"{'Epocha':>10} | {'Train MSE':>12} | {'Train MAE':>12} | "
            if X_test is not None and y_test is not None:
                header += f"{'Test MSE':>12} | {'Test MAE':>12} | "
            header += f"{'a (sklon)':>12} | {'b (posun)':>12}"
            print(header)
            print("-" * len(header))

        for epoch in range(1, self.epochs + 1):
            # 1. Predikce na trénovací sadě (škálovaná doména)
            y_pred_z = self.w_z * z_tr + self.b_z
            error = y_pred_z - y_tr
            
            # 2. Výpočet gradientů MSE vzhledem k w_z a b_z
            # MSE = (1/n) * sum((w_z * z + b_z - y)^2)
            # d(MSE)/d(w_z) = (2/n) * sum(error * z)
            # d(MSE)/d(b_z) = (2/n) * sum(error)
            grad_w = (2.0 / n) * np.sum(error * z_tr)
            grad_b = (2.0 / n) * np.sum(error)
            
            # 3. Krok gradientního sestupu
            self.w_z -= self.learning_rate * grad_w
            self.b_z -= self.learning_rate * grad_b
            
            # Aktualizace reálných parametrů a, b
            self._update_unscaled_weights()
            
            # 4. Výpočet metrik
            train_mse = float(np.mean(error ** 2))
            train_mae = float(np.mean(np.abs(error)))
            
            test_mse = None
            test_mae = None
            if X_test is not None and y_test is not None:
                y_test_pred = self.predict(X_test)
                test_mse = float(np.mean((y_test - y_test_pred) ** 2))
                test_mae = float(np.mean(np.abs(y_test - y_test_pred)))
            
            # Uložení do historie
            entry = {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "test_mse": test_mse,
                "test_mae": test_mae,
                "a": self.a,
                "b": self.b,
            }
            self.history.append(entry)
            
            # 5. Periodický výpis průběhu do terminálu
            if verbose and (epoch == 1 or epoch % self.log_interval == 0 or epoch == self.epochs):
                line = f"{epoch:>10d} | {train_mse:>12.6f} | {train_mae:>12.6f} | "
                if test_mse is not None and test_mae is not None:
                    line += f"{test_mse:>12.6f} | {test_mae:>12.6f} | "
                line += f"{self.a:>12.6f} | {self.b:>12.6f}"
                print(line)
                
            if step_delay > 0:
                time.sleep(step_delay)
                
        if verbose:
            print("=" * 105)
            print(f" Trénování dokončeno! Finální model: f(x) = {self.a:.6f} * x + {self.b:.6f}")
            print(f" Porovnání s OLS optimem:     f(x) = {self.ols_a:.6f} * x + {self.ols_b:.6f}")
            print("=" * 105)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Provede predikci cen na základě zadané plochy X: f(x) = a * x + b.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        return self.a * x_arr + self.b

    def get_weights(self) -> Dict[str, Any]:
        """Vrátí slovník s naučenými vahami a tvarem modelu."""
        return {
            "a": float(self.a),
            "b": float(self.b),
            "equation": f"f(x) = {self.a:.6f} * x + {self.b:.6f}",
            "feature": self.feature_name,
            "target": self.target_name,
            "units": {self.feature_name: "m2" if self.feature_name == "plocha" else "jednotka", self.target_name: "mil. Kč"},
            "ols_reference": {
                "a": float(self.ols_a) if self.ols_a is not None else None,
                "b": float(self.ols_b) if self.ols_b is not None else None,
                "equation": f"f(x) = {self.ols_a:.6f} * x + {self.ols_b:.6f}" if self.ols_a is not None else None,
            }
        }

    def save(self, model_pkl_path: str, weights_json_path: str, model_json_path: Optional[str] = None) -> None:
        """Uloží model (pickle) a váhy (json)."""
        # 1. Uložení binárního modelu
        with open(model_pkl_path, "wb") as f:
            pickle.dump(self, f)
            
        # 2. Uložení vah do JSON
        weights = self.get_weights()
        with open(weights_json_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=4, ensure_ascii=False)
            
        # 3. Uložení JSON reprezentace modelu
        if model_json_path:
            model_info = {
                "model_type": "LinearRegressionGD",
                "equation": f"f(x) = {self.a:.6f} * x + {self.b:.6f}",
                "weights": {"a": self.a, "b": self.b},
                "feature": self.feature_name,
                "target": self.target_name,
                "hyperparameters": {
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "log_interval": self.log_interval,
                },
                "scaling": {
                    "x_mean": self.x_mean,
                    "x_std": self.x_std,
                },
                "ols_optimum": {
                    "a": self.ols_a,
                    "b": self.ols_b,
                }
            }
            with open(model_json_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, model_pkl_path: str) -> "LinearRegressionGD":
        """Načte uložený model z pickle souboru."""
        with open(model_pkl_path, "rb") as f:
            model = pickle.load(f)
        return model


class MultipleLinearRegressionGD:
    """
    Vícenásobný lineární regresní model f(X) = X @ w + b pro více příznaků.
    (Např. kvalita, plocha, rok_stavby, rok_prodeje).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1000,
        log_interval: int = 100,
        feature_names: Optional[List[str]] = None,
        target_name: str = "cena",
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.log_interval = log_interval
        self.feature_names = feature_names or ["kvalita", "plocha", "rok_stavby", "rok_prodeje"]
        self.target_name = target_name

        self.w: np.ndarray = np.array([])
        self.b: float = 0.0

        self.means: np.ndarray = np.array([])
        self.stds: np.ndarray = np.array([])

        self.w_z: np.ndarray = np.array([])
        self.b_z: float = 0.0

        self.history: List[Dict[str, Any]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "MultipleLinearRegressionGD":
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64).flatten()
        n, d = X_tr.shape

        self.means = np.mean(X_tr, axis=0)
        self.stds = np.std(X_tr, axis=0)
        self.stds[self.stds == 0.0] = 1.0

        Z_tr = (X_tr - self.means) / self.stds

        self.w_z = np.zeros(d, dtype=np.float64)
        self.b_z = float(np.mean(y_tr))
        self.history = []

        if verbose:
            print("=" * 105)
            print(" Zahájení trénování vícenásobného lineárního modelu f(X) = X @ w + b (Gradient Descent)")
            print(f" Příznaky: {self.feature_names}")
            print(f" Dataset: {n} trénovacích vzorků" + (f", {len(X_test)} testovacích" if X_test is not None else ""))
            print(f" Hyperparametry: lr={self.learning_rate}, epochs={self.epochs}, log_interval={self.log_interval}")
            print("=" * 105)

        for epoch in range(1, self.epochs + 1):
            y_pred_z = Z_tr @ self.w_z + self.b_z
            error = y_pred_z - y_tr

            grad_w = (2.0 / n) * (Z_tr.T @ error)
            grad_b = (2.0 / n) * np.sum(error)

            self.w_z -= self.learning_rate * grad_w
            self.b_z -= self.learning_rate * grad_b

            # Převedení vah do původní škály
            self.w = self.w_z / self.stds
            self.b = float(self.b_z - np.sum(self.w_z * self.means / self.stds))

            train_mse = float(np.mean(error ** 2))
            train_mae = float(np.mean(np.abs(error)))

            test_mse = None
            test_mae = None
            if X_test is not None and y_test is not None:
                y_test_pred = self.predict(X_test)
                test_mse = float(np.mean((y_test - y_test_pred) ** 2))
                test_mae = float(np.mean(np.abs(y_test - y_test_pred)))

            self.history.append({
                "epoch": epoch,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "test_mse": test_mse,
                "test_mae": test_mae,
            })

            if verbose and (epoch == 1 or epoch % self.log_interval == 0 or epoch == self.epochs):
                line = f"Epocha {epoch:>6d} | Train MSE: {train_mse:>10.6f} | Train MAE: {train_mae:>10.6f}"
                if test_mse is not None:
                    line += f" | Test MSE: {test_mse:>10.6f} | Test MAE: {test_mae:>10.6f}"
                print(line)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return X_arr @ self.w + self.b

    def get_weights(self) -> Dict[str, Any]:
        terms = [f"{w:.6f} * {name}" for w, name in zip(self.w, self.feature_names)]
        eq = " + ".join(terms) + f" + {self.b:.6f}"
        return {
            "w": self.w.tolist(),
            "b": float(self.b),
            "equation": f"f(X) = {eq}",
            "features": self.feature_names,
            "target": self.target_name,
        }

    def save(self, model_pkl_path: str, weights_json_path: str) -> None:
        with open(model_pkl_path, "wb") as f:
            pickle.dump(self, f)
        with open(weights_json_path, "w", encoding="utf-8") as f:
            json.dump(self.get_weights(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, model_pkl_path: str) -> "MultipleLinearRegressionGD":
        with open(model_pkl_path, "rb") as f:
            return pickle.load(f)

