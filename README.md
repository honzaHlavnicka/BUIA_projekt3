# Projekt 3: Jednoduchý lineární model pro predikci cen bytů

Tento projekt implementuje a trénuje **jednoduchý lineární regresní model** ve tvaru:
$$f(x) = a \cdot x + b$$
pro predikci ceny nemovitostí v milionech Kč na základě užitné plochy v $m^2$.

Řešení je vytvořeno v jazyce **Python** (s využitím knihoven `numpy`, `pandas` a `matplotlib`) a optimalizováno pomocí algoritmu **Gradient Descent (gradientní sestup)** s průběžným výpisem trénovacího procesu a přesnosti do terminálu.

Na řešení byly použit AI nástroje.
---

## 1. Zadání a splněné požadavky

* **Dataset**: Dataset `ceny_domu.csv` (450 řádků) rozdělený na trénovací a testovací sadu v poměru **9:1** (405 trénovacích, 45 testovacích vzorků).
* **Model**: Jednoduchý lineární model $f(x) = ax + b$ (plocha $\rightarrow$ cena). Jako rozšíření je implementován i vícenásobný model se všemi příznaky (`kvalita`, `plocha`, `rok_stavby`, `rok_prodeje`).
* **Průběžný výpis v terminálu**: V průběhu trénování se v nastavených intervalech vypisují hodnoty Train MSE, Train MAE, Test MSE, Test MAE a aktuální parametry $a$ a $b$.
* **Uložení výstupů**:
  * Váhy modelu ($a$, $b$): `output/vahy.json`
  * Serializovaný model: `output/model.pkl` a `output/model.json`
  * Statistika chyb (MSE, MAE, R²): `output/statistiky.json`
  * Graf vývoje chyby (učící křivka): `output/learning_curve.png`
  * Graf regresní přímky proložené daty: `output/regression_fit.png`
* **Validace a predikce**: Skript `predict.py` pro predikci zadané plochy, validaci ze souboru (např. zadaná struktura `kvalita,plocha,rok_stavby,rok_prodeje,cena`) nebo interaktivní režim.

---

## 2. Matematický princip a odvození

### 2.1 Model a chybová funkce
Pro $N$ vzorků s plochou $x_i$ a skutečnou cenou $y_i$ je predikovaná cena:
$$\hat{y}_i = f(x_i) = a \cdot x_i + b$$

Chyba modelu je měřena pomocí střední kvadratické chyby (**Mean Squared Error – MSE**):
$$\mathcal{L}_{MSE}(a, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 = \frac{1}{N} \sum_{i=1}^{N} (a \cdot x_i + b - y_i)^2$$

Doplňkovou metrikou je průměrná absolutní chyba (**Mean Absolute Error – MAE**):
$$\mathcal{L}_{MAE}(a, b) = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$

### 2.2 Gradientní sestup (Gradient Descent)
V každé epoše se parametry $a$ (sklon) a $b$ (posun / intercept) aktualizují proti směru gradientu funkce MSE:
$$\frac{\partial \mathcal{L}_{MSE}}{\partial a} = \frac{2}{N} \sum_{i=1}^{N} (a \cdot x_i + b - y_i) \cdot x_i$$

$$\frac{\partial \mathcal{L}_{MSE}}{\partial b} = \frac{2}{N} \sum_{i=1}^{N} (a \cdot x_i + b - y_i)$$

Pravidlo aktualizace s rychlostí učení $\alpha$ (learning rate):
$$a^{(t+1)} = a^{(t)} - \alpha \cdot \frac{\partial \mathcal{L}_{MSE}}{\partial a}$$
$$b^{(t+1)} = b^{(t)} - \alpha \cdot \frac{\partial \mathcal{L}_{MSE}}{\partial b}$$

Pro zajištění stability a rychlé konvergence jsou vstupní hodnoty $x$ vnitřně standardizovány ($z = \frac{x - \mu_x}{\sigma_x}$), přičemž parametry jsou v každém kroku analyticky přepočítávány zpět do původního měřítka, takže výsledná rovnice má přímou fyzikální interpretaci:
$$a = \frac{w_z}{\sigma_x}, \quad b = b_z - a \cdot \mu_x$$

### 2.3 Kontrola optimality (Metoda nejmenších čtverců - OLS)
Trénovací algoritmus zároveň počítá teoretické analytické optimum (Ordinary Least Squares):
$$a_{ols} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b_{ols} = \bar{y} - a_{ols} \bar{x}$$
Naučené váhy algoritmu Gradient Descent přesně konvergují k tomuto optimu.

---

## 3. Struktura projektu

```
projekt3/
├── data/
│   ├── ceny_domu.csv              # Originální dataset (450 záznamů)
│   └── validace_vzor.csv          # Šablona pro validaci (kvalita,plocha,rok_stavby,rok_prodeje,cena)
├── src/
│   ├── __init__.py
│   ├── model.py                   # Třídy LinearRegressionGD a MultipleLinearRegressionGD
│   ├── utils.py                   # Načtení dat, rozdělení 9:1, výpočty MSE, MAE, R²
│   └── visualizer.py              # Generování grafů (learning_curve, regression_fit)
├── output/                        # Generované výstupy
│   ├── vahy.json                  # Uložené váhy a a b (+ OLS reference)
│   ├── model.pkl                  # Serializovaný binární model
│   ├── model.json                 # Textová JSON specifikace modelu
│   ├── statistiky.json            # Metriky Train/Test MSE, MAE, R²
│   ├── train_data.csv             # 405 trénovacích vzorků (90 %)
│   ├── test_data.csv              # 45 testovacích vzorků (10 %)
│   ├── learning_curve.png         # Graf vývoje chyb v čase
│   ├── regression_fit.png         # Graf regresní přímky v datech
│   └── validace_predikce.csv      # Výsledky validace s predikovanou cenou
├── train.py                       # Hlavní trénovací skript
├── predict.py                     # Skript pro predikci a validaci
└── README.md                      # Tato dokumentace
```

---

## 4. Návod k použití

### 4.1 Spuštění trénování

Základní spuštění (trénuje jednoduchý lineární model $f(x) = ax + b$ pro plochu):
```bash
python3 train.py
```

Spuštění s vlastními hyperparametry (např. 500 epoch, interval výpisu po 50 epochách, učící rychlost 0.05):
```bash
python3 train.py --epochs 500 --lr 0.05 --log-interval 50
```

Ukázka postupného výpisu do terminálu během trénování:
```
=========================================================================================================
 Zahájení trénování jednoduchého lineárního modelu f(x) = a * x + b (Gradient Descent)
 Dataset: 405 trénovacích vzorků, 45 testovacích vzorků
 Hyperparametry: learning_rate=0.05, epochs=300, log_interval=50
 Analytické OLS optimum: f(x) = 0.031650 * x + 0.105805
=========================================================================================================
    Epocha |    Train MSE |    Train MAE |     Test MSE |     Test MAE |    a (sklon) |    b (posun)
----------------------------------------------------------------------------------------------------
         1 |     3.616223 |     1.431956 |     4.344013 |     1.577218 |     0.003165 |     4.081914
        50 |     1.612947 |     0.915668 |     1.559877 |     0.884079 |     0.031487 |     0.128574
       100 |     1.612881 |     0.916647 |     1.559756 |     0.883075 |     0.031649 |     0.105923
       150 |     1.612881 |     0.916653 |     1.559755 |     0.883070 |     0.031650 |     0.105806
       200 |     1.612881 |     0.916653 |     1.559755 |     0.883070 |     0.031650 |     0.105805
       250 |     1.612881 |     0.916653 |     1.559755 |     0.883070 |     0.031650 |     0.105805
       300 |     1.612881 |     0.916653 |     1.559755 |     0.883070 |     0.031650 |     0.105805
=========================================================================================================
 Trénování dokončeno! Finální model: f(x) = 0.031650 * x + 0.105805
 Porovnání s OLS optimem:     f(x) = 0.031650 * x + 0.105805
=========================================================================================================
```

Pokud chcete natrénovat i vícenásobnou regresi se všemi příznaky:
```bash
python3 train.py --all-features
```

---

### 4.2 Predikce a validace (`predict.py`)

#### A. Predikce pro konkrétní plochu bytu:
```bash
python3 predict.py --plocha 120
```
Výstup:
```
 Predikce pro plochu 120.0 m²:
   Model:  f(x) = 0.031650 * x + 0.105805
   Vzorec: cena = 0.0317 * 120.0 + 0.1058
   Odhadovaná cena: 3.904 mil. Kč (3,903,829 Kč)
```

#### B. Hromadná validace ze souboru CSV:
```bash
python3 predict.py --file data/validace_vzor.csv
```
Výsledky se vypíší do tabulky a uloží do `output/validace_predikce.csv`.

#### C. Interaktivní konzolový režim:
```bash
python3 predict.py --interactive
```

---

## 5. Výsledky natrénovaného modelu

### Finální parametry modelu ($f = ax + b$):
* **Sklon ($a$)**: `0.031650` (každý $1\,m^2$ plochy zvyšuje odhadovanou cenu o cca 31 650 Kč)
* **Posun ($b$)**: `0.105805` (základní posun 105 805 Kč)
* **Rovnice modelu**:
  $$cena = 0.031650 \cdot plocha + 0.105805$$

### Dosažené metriky:
| Metrika | Trénovací sada (90 %, 405 vzorků) | Testovací sada (10 %, 45 vzorků) |
|---|---|---|
| **MSE** | 1.612881 (mil. Kč)² | 1.559755 (mil. Kč)² |
| **MAE** | 0.916653 mil. Kč (~916 653 Kč) | 0.883070 mil. Kč (~883 070 Kč) |
| **RMSE** | 1.269993 mil. Kč | 1.248902 mil. Kč |
| **$R^2$ skóre** | 0.553987 | 0.682388 |
