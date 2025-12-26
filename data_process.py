import pandas as pd
import re
import numpy as np

# --- LISTS DEFINITIONS ---
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


# --- functions ---

def get_season(m):
    # splits months to 4 seasons
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def get_day_time(h):
    # splits hours to 4 day times
    if 6 <= h < 12:
        return "morning"
    elif 12 <= h < 18:
        return "afternoon"
    elif 18 <= h < 22:
        return "evening"
    else:
        return "night"


def get_law(text):
    # find what law was broken
    if not isinstance(text, str):
        return "OTHER"

    text = text.lower().strip()

    # 1. find the paragraph (e.g. 125c)
    # look for number or char on the beginning or after §
    paragraph_match = re.search(r'(?:§\s*|^)?(\d+[a-z]?)', text)
    if not paragraph_match:
        return "OTHER"

    # save the base (e.g. "125c")
    result = paragraph_match.group(1)

    # 2. look for char after the base (125c/1k) vs 125c/1f))
    # look for char a-z before ')'
    char_match = re.search(r'([a-z])\)', text)

    if char_match:
        result += f"/{char_match.group(1)}"

    return result


def get_prague_district(n):
    # returns prague districts
    n = str(n).title().strip()
    if n.startswith("Praha ") and len(n) <= 8:
        parts = n.split(" ")
        if len(parts) > 1 and parts[1].isdigit():
            return "Praha " + parts[1]
    return "Praha - Ostatní"


def extract_car_brand(text):
    # returns car brand

    if not isinstance(text, str) or text == "Neuvedeno":
        return "UNSPECIFIED"

    for car in CARS_LIST:
        if car.lower() in text.lower():
            return car
    return "OTHER"


# --- main function for processing ---
def process_data(df):
    df = df.copy()

    # converts time
    df["DATSK"] = pd.to_datetime(df["DATSK"])
    temp_times = pd.to_datetime(df['CASSK'], format='mixed', errors='coerce')

    # basic time types
    df["MONTH_NUM"] = df["DATSK"].dt.month
    df["SEASON"] = df["MONTH_NUM"].apply(get_season)
    df["HOUR"] = temp_times.dt.hour.fillna(-1).astype(int)
    df["DAY_TIME"] = df["HOUR"].apply(get_day_time)
    """
    df["MISSING_TIME"] = (df["HOUR"] == -1).astype(int)
    """

    # prague district
    df["PRAGUE"] = df["PRAHA"].apply(get_prague_district)

    # car brand
    df["CAR_TYPE"] = df["TOVZN"].apply(extract_car_brand)
    """
    df["MISSING_BRAND"] = (df["CAR_TYPE"] == "UNSPECIFIED").astype(int)
    """

    # law
    df["LAW_CLEAN"] = df["PRAVFOR"].apply(get_law)

    # car owner is person or company (true if company)
    df["IS_FIRM"] = (df["FIRMA"] == "ANO").astype(int)

    # column list that goes to the training
    """
    cols_to_keep = [
        "SEASON", "DAY_TIME", "MISSING_TIME", "PRAGUE",
        "CAR_TYPE", "MISSING_BRAND", "LAW_CLEAN", "IS_FIRM"
    ]
    """

    cols_to_keep = [
        "SEASON", "DAY_TIME", "PRAGUE",
        "CAR_TYPE", "LAW_CLEAN", "IS_FIRM"
    ]

    # we need "OZNAM" column for training
    if "OZNAM" in df.columns:
        return df[cols_to_keep + ["OZNAM"]]
    else:
        return df[cols_to_keep]