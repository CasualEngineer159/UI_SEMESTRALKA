import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- KONFIGURACE A KONSTANTY ---

MAIN_STREETS_LIST = [
    "Evropská", "Plzeňská", "Strakonická", "Jižní spojka", "Štěrboholská",
    "5. května", "Wilsonova", "Argentinská", "Chodovská", "Liberecká",
    "Karlovarská", "Sokolovská", "Poděbradská", "Průmyslová", "Veleslavínská",
    "D1", "D0", "D5", "D8", "D10", "D11", "okruh", "spojka", "radiála"
]

CARS_LIST = [
    "Škoda", "Volkswagen", "Hyundai", "Toyota", "Kia", "Peugeot", "Dacia",
    "Renault", "Ford", "Mercedes", "BMW", "Audi", "Volvo", "Opel", "Mazda",
    "Suzuki", "Fiat", "Citroën", "Seat", "Honda", "Nissan", "IVECO",
    "Land Rover", "Porsche", "Mitsubishi"
]


# --- POMOCNÉ FUNKCE (Feature Engineering) ---
# Tyto funkce jsou nyní logicky identické s data_process.py

def get_season(m):
    """Rozdělí měsíce na 4 roční období."""
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def get_day_time(h):
    """Rozdělí hodiny na části dne."""
    if h == -1:
        return "none"
    elif 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "afternoon"
    elif 18 <= h < 22:
        return "evening"
    else:
        return "night"


def is_workday(weekday):
    """Určí, zda je den pracovní (0-4) nebo víkend (5-6)."""
    # Poznámka: weekday z pandas je 0=Pondělí, 6=Neděle
    if weekday in range(0, 5):
        return 1  # Pracovní den
    else:
        return 0  # Víkend


def get_place_type(text):
    """Určí typ místa: náměstí, tunel, hlavní tah, nebo ostatní."""
    if not isinstance(text, str):
        return "other"  # data_process vrací lowercase "other"

    text_lower = text.lower()

    # V data_process.py NENÍ odstraňování 'směr', proto ho zde také vynecháme,
    # aby byl výstup identický.

    if "náměstí" in text_lower:
        return "square"  # data_process vrací lowercase
    if "tunel" in text_lower:
        return "tunnel"  # data_process vrací lowercase

    for street in MAIN_STREETS_LIST:
        if street.lower() in text_lower:
            return "main_street"  # data_process vrací lowercase

    return "other"


def extract_car_brand(text):
    """Extrahuje značku auta nebo vrátí OTHER/UNSPECIFIED."""
    if not isinstance(text, str) or text == "Neuvedeno":
        return "UNSPECIFIED"

    for car in CARS_LIST:
        if car.lower() in text.lower():
            return car
    return "OTHER"


def get_prague_district(n):
    """Formátuje městskou část."""
    n = str(n).title().strip()
    if n.startswith("Praha ") and len(n) <= 8:
        parts = n.split(" ")
        if len(parts) > 1 and parts[1].isdigit():
            return "Praha " + parts[1]
    return "Praha - Ostatní"


def get_law(text):
    """Parsuje porušený zákon (paragraf/odstavec/písmeno)."""
    if not isinstance(text, str):
        return "OTHER"

    text = text.lower().strip()

    # Hledání paragrafu (např. 125c)
    paragraph_match = re.search(r'(?:§\s*|^)?(\d+[a-z]?)', text)
    if not paragraph_match:
        return "OTHER"

    result = paragraph_match.group(1)

    # Hledání písmene v závorce
    char_match = re.search(r'([a-z])\)', text)
    if char_match:
        result += f"/{char_match.group(1)}"

    return result


def encode_column_for_corr(series):
    """
    Jednoduchý Label Encoding pro účely korelační matice.
    """
    series = series.astype(str)
    labels, uniques = pd.factorize(series)
    return labels, uniques


# --- HLAVNÍ FUNKCE ---

def complex_data_analysis(df, show_plot=True):

    print("Zpracovávám data...")

    # 2. ZÁKLADNÍ ZPRACOVÁNÍ (IDENTICKÉ S data_process.py)
    # Musíme dodržet přesný postup, aby seděly typy a hodnoty

    # Čas
    df["DATSK"] = pd.to_datetime(df["DATSK"])
    # Použijeme format='mixed' a coerce, stejně jako v data_process
    temp_times = pd.to_datetime(df['CASSK'], format='mixed', errors='coerce')

    # Základní features vyžadované pro trénink
    df["MONTH_NUM"] = df["DATSK"].dt.month
    df["SEASON"] = df["MONTH_NUM"].apply(get_season)

    # Hodina: fillna(-1) a int, přesně jako v data_process
    df["HOUR"] = temp_times.dt.hour.fillna(-1).astype(int)
    df["DAY_TIME"] = df["HOUR"].apply(get_day_time)

    df["PRAGUE"] = df["PRAHA"].apply(get_prague_district)
    df["PLACE_TYPE"] = df["MISTOSK"].apply(get_place_type)
    df["CAR_TYPE"] = df["TOVZN"].apply(extract_car_brand)
    df["LAW_CLEAN"] = df["PRAVFOR"].apply(get_law)
    df["IS_FIRM"] = (df["FIRMA"] == "ANO").astype(int)

    # 3. ROZŠÍŘENÁ ANALÝZA (NAVÍC OPROTI data_process.py)
    # Vytváříme sloupce navíc pouze pro účely korelační matice.
    # Tyto sloupce nebudou ve finálním výstupu pro model.

    df["YEAR"] = df["DATSK"].dt.year
    df["WEEKDAY"] = df["DATSK"].dt.weekday
    df["WORKDAY"] = df["WEEKDAY"].apply(is_workday)

    # Očištění země původu (jen pro korelaci, v raw datech necháváme původní)
    df["COUNTRY"] = df["MPZ"].fillna("UNKNOWN")

    if "OZNAM" in df.columns:
        df["WHO_RAW"] = df["OZNAM"]

    # 4. KORELAČNÍ ANALÝZA
    # Vytvoříme kopii pro enkódování
    df_encoded = df.copy()

    print("Počítám korelační matici ze všech dostupných příznaků...")

    # Pro vizualizaci korelace sloučíme málo četné kategorie do "OTHER",
    # aby matice nebyla rozbitá (toto se nepropíše do finálních dat)
    cols_to_simplify = ["CAR_TYPE", "LAW_CLEAN", "COUNTRY"]
    for col in cols_to_simplify:
        counts = df_encoded[col].value_counts(normalize=True)
        main_vals = counts[counts > 0.001].index
        df_encoded[col] = df_encoded[col].where(df_encoded[col].isin(main_vals), "OTHER")

    cols_to_encode = [
        "SEASON", "DAY_TIME", "PRAGUE", "PLACE_TYPE",
        "CAR_TYPE", "LAW_CLEAN", "COUNTRY", "WHO_RAW"
    ]

    for col in cols_to_encode:
        if col in df_encoded.columns:
            df_encoded[col], _ = encode_column_for_corr(df_encoded[col])

    # Výběr všech zajímavých sloupců pro matici
    corr_cols = [
        "YEAR", "MONTH_NUM", "HOUR", "WORKDAY", "SEASON", "DAY_TIME",
        "PRAGUE", "PLACE_TYPE", "COUNTRY", "CAR_TYPE", "IS_FIRM",
        "LAW_CLEAN", "WHO_RAW"
    ]
    print(df_encoded.columns)
    corr_cols = [c for c in corr_cols if c in df_encoded.columns]

    corr_matrix = df_encoded[corr_cols].corr()

    if show_plot:
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot_kws={"size": 14}  # Zvětšení písma hodnot v matici
        )
        plt.title("Korelační matice: Všechny extrahované příznaky", fontsize=20)  # Zvětšení nadpisu
        plt.xticks(fontsize=14, rotation=45)  # Zvětšení a natočení popisků osy X
        plt.yticks(fontsize=14, rotation=0)  # Zvětšení popisků osy Y
        plt.tight_layout()
        plt.savefig("correlation_matrix.png")
        plt.show()

    # 5. VÝBĚR FINÁLNÍCH SLOUPCŮ (SHODA S data_process.py)
    print("Vybírám finální sloupce pro trénink...")

    cols_to_keep = [
        "SEASON", "DAY_TIME", "PRAGUE", "PLACE_TYPE",
        "CAR_TYPE", "LAW_CLEAN", "IS_FIRM"
    ]

    if "OZNAM" in df.columns:
        final_cols = cols_to_keep + ["OZNAM"]
    else:
        final_cols = cols_to_keep

    # Vracíme df[final_cols], což zaručuje shodu s data_process.py
    df_final = df[final_cols].copy()

    return df_final


if __name__ == "__main__":
    # Test shody
    try:
        print("--- SPUŠTĚNÍ COMPREHENSIVE ---")
        raw_df = pd.read_csv("MHMP_dopravni_prestupky_2024.csv")
        my_df = complex_data_analysis(raw_df)

        print("\n--- SPUŠTĚNÍ DATA_PROCESS ---")
        from data_process import process_data


        ref_df = process_data(raw_df)

        print(f"\nRozměry Comprehensive: {my_df.shape}")
        print(f"Rozměry Data Process:  {ref_df.shape}")

        # Kontrola shody sloupců
        diff = my_df.compare(ref_df)
        if diff.empty:
            print("\n✅ VÝSTUPY JSOU IDENTICKÉ!")
        else:
            print("\n❌ NALEZENY ROZDÍLY:")
            print(diff.head())

    except FileNotFoundError:
        print("CSV soubor nenalezen.")