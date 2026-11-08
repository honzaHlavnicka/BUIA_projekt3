#!/usr/bin/env python3
"""
Predikční a validační skript pro natrénovaný model f(x) = a * x + b.

Umožňuje:
1. Predikci pro zadanou plochu (--plocha 120)
2. Validaci a predikci ze souboru CSV (--file data/validace_vzor.csv)
3. Interaktivní zadání plochy přes konzoli
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

from src.model import LinearRegressionGD, MultipleLinearRegressionGD
from src.utils import mean_squared_error, mean_absolute_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predikce ceny bytů pomocí natrénovaného lineárního modelu"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="output/model.pkl",
        help="Cesta k uloženému modelu (výchozí: output/model.pkl)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="output/vahy.json",
        help="Alternativní načtení přímo z JSONu s vahami (výchozí: output/vahy.json)",
    )
    parser.add_argument(
        "--plocha",
        type=float,
        default=None,
        help="Obytná plocha v m² pro okamžitou predikci (např. --plocha 120)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Cesta k CSV souboru pro validaci/predikci (např. data/validace_vzor.csv)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/validace_predikce.csv",
        help="Kam uložit výsledný CSV soubor s doplněnými predikcemi (výchozí: output/validace_predikce.csv)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Spustí interaktivní konzolový režim pro zadávání ploch",
    )
    return parser.parse_args()


def load_trained_model(model_path: str, weights_path: str):
    """Pokusí se načíst model z pickle, případně z JSONu s vahami."""
    if os.path.exists(model_path):
        try:
            import pickle
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Varování: Nepodařilo se načíst {model_path} ({e}), zkouším JSON...")

    if os.path.exists(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "w" in data:
            model = MultipleLinearRegressionGD(feature_names=data.get("features"))
            model.w = np.array(data["w"], dtype=np.float64)
            model.b = float(data["b"])
            return model
        else:
            model = LinearRegressionGD()
            model.a = float(data["a"])
            model.b = float(data["b"])
            return model

    raise FileNotFoundError(
        f"Model nebyl nalezen v '{model_path}' ani '{weights_path}'. Nejprve spusťte 'python3 train.py'!"
    )


def predict_single(model, plocha: float):
    cena = float(model.predict(np.array([plocha]))[0])
    print(f"\n Predikce pro plochu {plocha:.1f} m²:")
    print(f"   Model:  {model.get_weights()['equation']}")
    print(f"   Vzorec: cena = {model.a:.4f} * {plocha:.1f} + {model.b:.4f}")
    print(f"   Odhadovaná cena: {cena:.3f} mil. Kč ({cena * 1_000_000:,.0f} Kč)\n")
    return cena


def predict_csv(model, input_csv: str, output_csv: str):
    print(f"\n Načítám validační data z: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if hasattr(model, "feature_names") and len(model.feature_names) > 1:
        missing_cols = [c for c in model.feature_names if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Soubor {input_csv} neobsahuje sloupce potřebné pro vícenásobný model: {missing_cols}")
        X_mat = df[model.feature_names].to_numpy(dtype=np.float64)
        pred_ceny = model.predict(X_mat)
    else:
        if "plocha" not in df.columns:
            raise ValueError(f"Soubor {input_csv} neobsahuje povinný sloupec 'plocha'!")
        plochy = df["plocha"].to_numpy(dtype=np.float64)
        pred_ceny = model.predict(plochy)
    
    df["predikovana_cena"] = np.round(pred_ceny, 3)
    
    # Pokud jsou v původních datech známé skutečné ceny, spočítáme validační metriky
    valid_mask = df["cena"].notna() & (df["cena"] > 0)
    has_ground_truth = valid_mask.any()
    
    print("\n" + "=" * 80)
    print(" VÝSLEDKY PREDIKCE / VALIDACE")
    print("=" * 80)
    display_cols = [c for c in ["kvalita", "plocha", "rok_stavby", "rok_prodeje", "cena", "predikovana_cena"] if c in df.columns]
    print(df[display_cols].to_string(index=False))
    print("=" * 80)

    if has_ground_truth:
        y_true = df.loc[valid_mask, "cena"].to_numpy(dtype=np.float64)
        y_pred = df.loc[valid_mask, "predikovana_cena"].to_numpy(dtype=np.float64)
        val_mse = mean_squared_error(y_true, y_pred)
        val_mae = mean_absolute_error(y_true, y_pred)
        print(f" Validační metriky na vzorcích se známou cenou (n={len(y_true)}):")
        print(f"   -> Validační MSE: {val_mse:.4f} (mil. Kč)²")
        print(f"   -> Validační MAE: {val_mae:.4f} mil. Kč")
        print("=" * 80)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f" Výsledky s predikovanou cenou uloženy do: {output_csv}\n")


def interactive_mode(model):
    print("\n--- Interaktivní režim odhadu ceny bytu (pro ukončení zadejte 'q') ---")
    print(f"Použitý model: {model.get_weights()['equation']}\n")
    while True:
        try:
            user_input = input("Zadejte plochu bytu v m²: ").strip()
            if user_input.lower() in ("q", "quit", "konec", "exit"):
                print("Ukončuji.")
                break
            plocha = float(user_input)
            if plocha <= 0:
                print("Plocha musí být kladné číslo.")
                continue
            predict_single(model, plocha)
        except ValueError:
            print("Neplatná hodnota. Zadejte prosím číslo (např. 85.5).")
        except (KeyboardInterrupt, EOFError):
            print("\nUkončeno.")
            break


def main():
    args = parse_args()
    try:
        model = load_trained_model(args.model, args.weights)
    except FileNotFoundError as err:
        print(f"\n Chyba: {err}")
        sys.exit(1)

    if args.plocha is not None:
        predict_single(model, args.plocha)
    elif args.file is not None:
        predict_csv(model, args.file, args.output_file)
    elif args.interactive:
        interactive_mode(model)
    else:
        # Výchozí ukázka pokud nejsou předány parametry
        print(f" Načten natrénovaný model: {model.get_weights()['equation']}")
        print(" Provádím ukázkovou predikci pro typické plochy bytů:")
        for demo_plocha in [45.0, 75.0, 100.0, 150.0]:
            demo_pred = float(model.predict(np.array([demo_plocha]))[0])
            print(f"   - {demo_plocha:>5.1f} m²  -->  {demo_pred:>6.3f} mil. Kč  ({demo_pred*1_000_000:>10,.0f} Kč)")
        print("\nTip: Spusťte s argumentem --plocha X, --file data/validace_vzor.csv nebo --interactive")


if __name__ == "__main__":
    main()
