#!/usr/bin/env python3
"""
Trénovací skript pro jednoduchý lineární model f(x) = a * x + b pro predikci ceny bytů.

Rozdělení dat 9:1, průběžný výpis do terminálu, uložení vah, modelu a statistik.
"""

import argparse
import json
import os
import sys

from src.utils import load_data, train_test_split_9_1, compute_metrics
from src.model import LinearRegressionGD, MultipleLinearRegressionGD
from src.visualizer import plot_learning_curves, plot_regression_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trénování jednoduchého lineárního modelu f(x) = ax + b (Gradient Descent)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ceny_domu.csv",
        help="Cesta k CSV souboru s daty (výchozí: data/ceny_domu.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Cílová složka pro uložení modelu, vah a statistik (výchozí: output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Počet trénovacích epoch (výchozí: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Rychlost učení / learning rate (výchozí: 0.05)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval výpisu přesnosti do terminálu v epochách (výchozí: 100)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Prodleva mezi epochami v sekundách pro sledování v reálném čase (výchozí: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Náhodné semínko pro reprodukovatelnost rozdělení 9:1 (výchozí: 42)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Poměr testovacích dat, 0.1 odpovídá rozdělení 9:1 (výchozí: 0.1)",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="plocha",
        help="Vstupní příznak pro jednoduchý model f(x) = ax + b (výchozí: plocha)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Trénuje také vícenásobnou lineární regresi se všemi příznaky (kvalita, plocha, rok_stavby, rok_prodeje)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Kontrola a načtení datasetu
    if not os.path.exists(args.data):
        print(f"Soubor {args.data} nenalezen, pokusím se stáhnout...")
        import urllib.request
        url = "https://raw.githubusercontent.com/mlcollege/ai-academy/main/06-Regrese/data/ceny_domu.csv"
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        urllib.request.urlretrieve(url, args.data)
        print(f"Dataset úspěšně stažen do {args.data}")

    print(f" Načítám data ze souboru: {args.data}")
    df = load_data(args.data)
    total_samples = len(df)
    print(f" Celkem načteno {total_samples} řádků se sloupci: {list(df.columns)}")

    # 2. Rozdělení na trénovací a testovací data v poměru 9:1
    X_train, y_train, X_test, y_test, df_train, df_test = train_test_split_9_1(
        df,
        target_col="cena",
        feature_col=args.feature,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )
    print(f" Rozdělení dat v poměru 9:1:")
    print(f"   -> Trénovací data: {len(X_train)} vzorků ({len(X_train)/total_samples*100:.1f} %)")
    print(f"   -> Testovací data:  {len(X_test)} vzorků ({len(X_test)/total_samples*100:.1f} %)")

    # Uložení rozdělených dat pro transparentnost
    df_train.to_csv(os.path.join(args.output_dir, "train_data.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test_data.csv"), index=False)

    # 3. Inicializace a trénování modelu s periodickým výpisem do terminálu
    model = LinearRegressionGD(
        learning_rate=args.lr,
        epochs=args.epochs,
        log_interval=args.log_interval,
        feature_name=args.feature,
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        step_delay=args.delay,
        verbose=True,
    )

    # 4. Vyhodnocení výsledných metrik na trénovací i testovací sadě
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print("\n" + "=" * 55)
    print(" SOUHRN VÝSLEDKŮ A METRIK NATRÉNOVANÉHO MODELU")
    print("=" * 55)
    print(f" Model:               {model.get_weights()['equation']}")
    print(f" Trénovací MSE:       {train_metrics['mse']:.6f} (mil. Kč)²")
    print(f" Trénovací MAE:       {train_metrics['mae']:.6f} mil. Kč")
    print(f" Trénovací R² skóre:  {train_metrics['r2']:.6f}")
    print("-" * 55)
    print(f" Testovací MSE:       {test_metrics['mse']:.6f} (mil. Kč)²")
    print(f" Testovací MAE:       {test_metrics['mae']:.6f} mil. Kč")
    print(f" Testovací R² skóre:  {test_metrics['r2']:.6f}")
    print("=" * 55)

    # 5. Uložení vah, modelu a statistik
    weights_path = os.path.join(args.output_dir, "vahy.json")
    model_pkl_path = os.path.join(args.output_dir, "model.pkl")
    model_json_path = os.path.join(args.output_dir, "model.json")
    stats_path = os.path.join(args.output_dir, "statistiky.json")

    model.save(
        model_pkl_path=model_pkl_path,
        weights_json_path=weights_path,
        model_json_path=model_json_path,
    )

    statistics_data = {
        "model_equation": f"f(x) = {model.a:.6f} * x + {model.b:.6f}",
        "feature": "plocha",
        "target": "cena",
        "dataset_split": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "ratio": "9:1",
            "seed": args.seed,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "weights": {
            "a_sklon": float(model.a),
            "b_posun": float(model.b),
        },
        "ols_optimal_reference": {
            "a": float(model.ols_a),
            "b": float(model.ols_b),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "log_interval": args.log_interval,
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics_data, f, indent=4, ensure_ascii=False)

    print(f"\n Uloženo:")
    print(f"   -> Váhy modelu:    {weights_path}")
    print(f"   -> Model (pickle): {model_pkl_path}")
    print(f"   -> Model (JSON):   {model_json_path}")
    print(f"   -> Statistiky:     {stats_path}")

    # 6. Vygenerování grafů
    curve_path = os.path.join(args.output_dir, "learning_curve.png")
    plot_learning_curves(model.history, curve_path)

    fit_path = os.path.join(args.output_dir, "regression_fit.png")
    plot_regression_fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        a=model.a,
        b=model.b,
        test_metrics=test_metrics,
        output_path=fit_path,
    )

    # 7. Pokud je požadován vícenásobný model se všemi příznaky
    if args.all_features:
        feature_cols = ["kvalita", "plocha", "rok_stavby", "rok_prodeje"]
        X_train_m, y_train_m, X_test_m, y_test_m, _, _ = train_test_split_9_1(
            df,
            target_col="cena",
            feature_col=feature_cols,
            test_ratio=args.test_ratio,
            random_state=args.seed,
        )
        print("\n" + "=" * 55)
        print(" TRÉNOVÁNÍ VÍCENÁSOBNÉHO MODELU (SE VŠEMI PŘÍZNAKY)")
        print("=" * 55)
        model_m = MultipleLinearRegressionGD(
            learning_rate=args.lr,
            epochs=args.epochs,
            log_interval=args.log_interval,
            feature_names=feature_cols,
        )
        model_m.fit(X_train_m, y_train_m, X_test_m, y_test_m, verbose=True)
        y_test_pred_m = model_m.predict(X_test_m)
        test_metrics_m = compute_metrics(y_test_m, y_test_pred_m)
        print("-" * 55)
        print(f" Vícenásobný model:   {model_m.get_weights()['equation']}")
        print(f" Testovací MSE:       {test_metrics_m['mse']:.6f} (mil. Kč)²")
        print(f" Testovací MAE:       {test_metrics_m['mae']:.6f} mil. Kč")
        print(f" Testovací R² skóre:  {test_metrics_m['r2']:.6f}")
        print("=" * 55)
        
        m_model_path = os.path.join(args.output_dir, "model_multivariate.pkl")
        m_weights_path = os.path.join(args.output_dir, "vahy_multivariate.json")
        m_stats_path = os.path.join(args.output_dir, "statistiky_multivariate.json")
        
        model_m.save(m_model_path, m_weights_path)
        with open(m_stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "equation": model_m.get_weights()["equation"],
                "features": feature_cols,
                "test_metrics": test_metrics_m,
                "weights": model_m.get_weights(),
            }, f, indent=4, ensure_ascii=False)
        print(f" Vícenásobný model uložen do: {m_model_path}")
        print(f" Váhy vícenásobného modelu:   {m_weights_path}")
        print(f" Statistiky vícenásobného:    {m_stats_path}")

    print("\n Hotovo! Pro predikci nových hodnot spusťte např.:")
    print(f"   python3 predict.py --plocha 120")
    print(f"   python3 predict.py --file data/validace_vzor.csv\n")


if __name__ == "__main__":
    main()
