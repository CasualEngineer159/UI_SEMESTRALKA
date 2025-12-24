import pprint as pp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from pandas import Series
from rich import print
import numpy as np
import re
from sklearn.utils import resample



# Načtení dat
MHMP = pd.read_csv("MHMP_dopravni_prestupky_2023.csv")
MHMP_copy = MHMP.copy()

# Doplnění chybějícího času
MHMP["CASSK"] = MHMP["CASSK"].fillna("00:00:00")

# Oprava dat s chybějícími sekundami
MHMP['CASSK'] = MHMP['CASSK'].apply(
    lambda t: t + ':00' if len(t) == 5 else t
)

# Převod na jeden sloupec s časem
MHMP["Time"] = MHMP["DATSK"] + " " + MHMP["CASSK"]
MHMP['Time'] = pd.to_datetime(MHMP['Time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Vyřazení dat s nečitelným formátem času
time_error_mask = pd.isna(MHMP["Time"])
MHMP = MHMP[~time_error_mask]

# Nastavení času jako index
MHMP.set_index("Time", inplace=True)

# Seřazení dle času
MHMP = MHMP.sort_index()

# Vytvoření čistého dataframe
data_clean = pd.DataFrame()
data_clean.index.name = 'Time'
data_clean["WHO"] = MHMP["OZNAM"]

# Rozložení času do sloupců
data_clean["YEAR"] = data_clean.index.year
data_clean["MONTH"] = data_clean.index.month
data_clean["DAY"] = data_clean.index.day
data_clean["HOUR"] = data_clean.index.hour

# Vytvoření sloupce všední den/víkend
data_clean["WORKDAY"] = data_clean.index.weekday
def is_workday(weekday):
    if weekday in range(0, 5):
        return 1
    else:
        return 0
data_clean["WORKDAY"] = data_clean["WORKDAY"].apply(is_workday)

# Pokud v původním záznamu nebyl čas, musí být hodina -1
missing_time_mask = (data_clean.index.hour == 0) & (data_clean.index.minute == 0) & (data_clean.index.second == 0)
data_clean.loc[missing_time_mask,"HOUR"] = -1

data_clean["PRAGUE"] = MHMP["PRAHA"].str.extract(r'(\d+)', expand=False)

data_clean["COUNTRY"] = MHMP["MPZ"]
data_clean["COUNTRY"] = data_clean["COUNTRY"].fillna("UNKNOWN")

counts = data_clean["COUNTRY"].value_counts(normalize=True)

threshold = 0.001

mask = counts > threshold
main_laws = counts[mask].index

# Všechno, co není v 'main_laws', nahradíme hodnotou 'OTHER'
data_clean["COUNTRY"] = data_clean["COUNTRY"].where(data_clean["COUNTRY"].isin(main_laws), "OTHER")


# Vymazání části stringu po slově směr
# Podolská směr Evropská -> Podolská
to_delete = r'\s*směr.*'

MHMP["MISTOSK"] = MHMP["MISTOSK"].str.replace(
    to_delete,
    '',
    regex=True,
    case=False
)

# Definice hlavních ulic
main_streets = [
    "Evropská", "Plzeňská", "Strakonická", "Jižní spojka", "Štěrboholská",
    "5. května", "Wilsonova", "Argentinská", "Chodovská", "Liberecká",
    "Karlovarská", "Sokolovská", "Poděbradská", "Průmyslová", "Veleslavínská",
    "D1", "D0", "D5", "D8", "D10", "D11", "okruh", "spojka", "radiála"
]
main_streets = '|'.join(main_streets)

# Určení typu místa
data_clean["PLACE"] = {}
square_mask = MHMP["MISTOSK"].str.contains("náměstí", case=False, na=False)
tunel_mask = MHMP["MISTOSK"].str.contains("tunel", case=False, na=False)
main_street_mask = MHMP["MISTOSK"].str.contains(main_streets, case=False, na=False)

# Vyplnění typu místa
data_clean["PLACE"] = "OTHER"
data_clean.loc[square_mask, "PLACE"] = "SQUARE"
data_clean.loc[tunel_mask, "PLACE"] = "TUNNEL"
data_clean.loc[main_street_mask, "PLACE"] = "MAIN_STREET"

# Definice značek aut
cars = [
    "Škoda", "Volkswagen", "Hyundai", "Toyota", "Kia",
    "Peugeot", "Dacia", "Renault", "Ford", "Mercedes",
    "BMW", "Audi", "Volvo", "Opel", "Mazda", "Suzuki", "Fiat", "Neuvedeno",
    "Citroën","Seat","Honda","Nissan","IVECO","Land Rover", "Porsche", "Mitsubishi",
    "Mini Cooper", "Lexus", "Subaru", "Chevrolet", "Jeep", "Citroen", "PASSAT"
]

data_clean["CAR_TYPE"] = {}
for car in cars:
    mask = MHMP["TOVZN"].str.contains(car, case=False, na=False)
    data_clean.loc[mask, "CAR_TYPE"] = car
mask_unspecified_car = (MHMP["TOVZN"] == "Neuvedeno") | (pd.isna(MHMP["TOVZN"]))
data_clean.loc[mask_unspecified_car, "CAR_TYPE"] = "UNSPECIFIED"
data_clean["CAR_TYPE"] = data_clean["CAR_TYPE"].fillna("OTHER")

mask_car_other = data_clean["CAR_TYPE"] == "OTHER"

# Spočítáme relativní četnosti (0.0 až 1.0)
counts = data_clean["CAR_TYPE"].value_counts(normalize=True)

# Stanovíme limit (např. 1 %, tedy 0.01)
threshold = 0.01

# Identifikujeme značky, které limit splňují
mask = counts > threshold
main_types = counts[mask].index

# Všechno, co není v 'main_types', nahradíme hodnotou 'OTHER'
data_clean["CAR_TYPE"] = data_clean["CAR_TYPE"].where(data_clean["CAR_TYPE"].isin(main_types), "OTHER")

# Rozdělení FIRMA/OSOBA
mask_person = MHMP["OSOBA"] == "ANO"
mask_firm = MHMP["FIRMA"] == "ANO"
data_clean["OSOBA"] = 0
data_clean["FIRMA"] = 0
data_clean.loc[mask_person, "OSOBA"] = 1
data_clean.loc[mask_firm, "FIRMA"] = 1

def sjednotit_zakon(text):
    if not isinstance(text, str):
        return pd.NA

    # 1. Základní očištění (malá písmena)
    text = text.lower().strip()

    # --- EXTRAKCE DAT POMOCÍ REGEXŮ ---

    # A) Paragraf (hledá číslo následované volitelně písmenem, např. 125c, 125, 16)
    # Ignorujeme znak § a hledáme hned na začátku nebo po mezeře
    paragraf_match = re.search(r'(?:§\s*|^)?(\d+[a-z]?)', text)
    paragraf = paragraf_match.group(1) if paragraf_match else ""

    # B) Odstavec (hledá číslo po "odst." nebo po lomítku "/")
    # Hledáme vzor: buď slovo 'odst' nebo lomítko, za ním volitelně mezera a číslice
    odstavec_match = re.search(r'(?:odst\.?|/)\s*(\d+)', text)
    odstavec = odstavec_match.group(1) if odstavec_match else ""

    # C) Písmeno (hledá písmeno a-z následované závorkou)
    # Může být po "písm.", nebo přímo nalepené za číslem (např. /1k)
    # Závorka ) je klíčový identifikátor pro písmeno v tomto datasetu
    pismeno_match = re.search(r'(?:písm\.?|[\d/])\s*([a-z])\)', text)
    pismeno = pismeno_match.group(1) if pismeno_match else ""

    # D) Bod (hledá číslo po slově "bod")
    bod_match = re.search(r'bod\s*(\d+)', text)
    bod = bod_match.group(1) if bod_match else ""

    # --- SESTAVENÍ VÝSLEDNÉHO FORMÁTU ---

    # Pokud nemáme ani paragraf, vrátíme původní text nebo chybu
    if not paragraf:
        return text

    vysledek = paragraf

    if odstavec:
        vysledek += f"/{odstavec}"

    if pismeno:
        vysledek += f"{pismeno})"

    if bod:
        # Přilepíme bod na konec (podle zadání "j)6").
        # Pokud byste chtěl mezeru, stačí změnit na f" {bod}"
        vysledek += f"{bod}"

    return vysledek

data_clean["LAW"] = MHMP["PRAVFOR"].apply(sjednotit_zakon)

# Spočítáme relativní četnosti (0.0 až 1.0)
counts = data_clean["LAW"].value_counts(normalize=True)

# Stanovíme limit (např. 1 %, tedy 0.01)
threshold = 0.001

# Identifikujeme zákony, které limit splňují
mask = counts > threshold
main_laws = counts[mask].index

# Všechno, co není v 'main_laws', nahradíme hodnotou 'OTHER'
data_clean["LAW"] = data_clean["LAW"].where(data_clean["LAW"].isin(main_laws), "OTHER")

df_legend = {}
def encode_column(column):
    legend = {}
    def encode(value):
        if value not in legend:
            legend[value] = len(legend)
        return legend[value]

    column = column.apply(encode)
    return column, legend

data_clean["LAW"], df_legend["LAW"] = encode_column(data_clean["LAW"])
data_clean["CAR_TYPE"], df_legend["CAR_TYPE"] = encode_column(data_clean["CAR_TYPE"])
data_clean["COUNTRY"], df_legend["COUNTRY"] = encode_column(data_clean["COUNTRY"])
data_clean["PLACE"], df_legend["PLACE"] = encode_column(data_clean["PLACE"])
data_clean["WHO"], df_legend["WHO"] = encode_column(data_clean["WHO"])

data_clean.to_csv("data_clean.csv")

data_clean = data_clean.drop(columns="YEAR")

def create_corr_matrix(df, name):
    # Výpočet korelační matice
    corr_matrix = df.corr()

    # Vizualizace pomocí heatmapy
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Korelační matice dopravních přestupků " + name)
    plt.show()

create_corr_matrix(data_clean, "VYČIŠTĚNÁ DATA")

# Rozdělení podle tříd
df_mpp = data_clean[data_clean.WHO == df_legend["WHO"]["MPP"]]  # Předpokládám klíč "MPP" nebo "MESTSKA..."
df_pcr = data_clean[data_clean.WHO == df_legend["WHO"]["PČR"]]  # Předpokládám klíč "PCR"

# Downsample většinové třídy (MPP)
df_mpp_downsampled = resample(df_mpp,
                              replace=False,  # nevracet vzorky
                              n_samples=len(df_pcr),  # počet jako menšinová třída
                              random_state=42)

# Spojení zpět
data_clean = pd.concat([df_mpp_downsampled, df_pcr])

create_corr_matrix(data_clean, "VYBALANCOVANÁ DATA (MPP : PČR = 1 : 1)")

# Zahození nepodstatných dat
data_clean = data_clean.drop(columns=["MONTH", "DAY", "HOUR", "WORKDAY"])

create_corr_matrix(data_clean, "VYBALANCOVÁNO + SMAZÁNY NEPODSTATNÉ SLOUPCE")