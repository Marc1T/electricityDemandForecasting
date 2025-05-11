import os
import time
import random
import yaml
import requests
import pandas as pd
import holidays

from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def load_config(config_path: str = "config.yaml") -> dict:
    """Lit config.yaml et remplace ${VARNAME} par la variable d'environnement."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    section[k] = os.getenv(v[2:-1])
    return cfg

def to_esios_date(iso_date: str, end_of_day: bool = False) -> str:
    """
    Convertit 'YYYY-MM-DD' en 'YYYY-MM-DDT00:00' ou 'YYYY-MM-DDT23:59'.
    """
    return f"{iso_date}T23:59" if end_of_day else f"{iso_date}T00:00"

def generate_periodic_windows(start_date: str, end_date: str, months: int = 1):
    """
    Génère des fenêtres successives de 'months' mois entre start_date et end_date.
    """
    windows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    final = datetime.strptime(end_date,   "%Y-%m-%d")
    while current <= final:
        nxt = current + relativedelta(months=months) - timedelta(days=1)
        if nxt > final:
            nxt = final
        windows.append((current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        current = nxt + timedelta(days=1)
    return windows

def fetch_data(indicator_id: int,
               start:        str,
               end:          str,
               token:        str,
               base_url:     str) -> pd.DataFrame:
    """
    Interroge ESIOS pour un indicateur, retourne un DataFrame avec colonnes
    ['datetime','geo_id','value'].
    """
    url = f"{base_url}/{indicator_id}"
    params = {
        "start_date": to_esios_date(start),
        "end_date":   to_esios_date(end, end_of_day=True)
    }
    headers = {
        "Accept":    "application/json; application/vnd.esios-api-v1+json",
        "x-api-key": token
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()["indicator"]["values"]
    df = pd.DataFrame(data)
    # Conserver la date UTC et la valeur
    df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df[["datetime", "geo_id", "value"]]

def save_hourly(df: pd.DataFrame, file_path: Path):
    """
    Sauvegarde df (index datetime, col value) en CSV, en mode append sans doublons.
    """
    df = df.set_index("datetime")[["value"]]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        existing = pd.read_csv(file_path,
                               parse_dates=["datetime"],
                               index_col="datetime")
        combined = pd.concat([existing, df]).drop_duplicates().sort_index()
        combined.to_csv(file_path, index_label="datetime")
    else:
        df.to_csv(file_path, index_label="datetime")

def process_national(cfg, start, split, raw_dir, months=3):
    """
    Récupère et sauvegarde la série nationale avant le split (2015-05 → 2021-06).
    """
    token    = cfg["esios"]["token"]
    base_url = cfg["esios"]["base_url"]
    ind_nat  = cfg["pvpc"]["national"]["indicator_id"]
    geo_nat  = cfg["pvpc"]["national"]["geo_id"]

    dfs = []
    for ws, we in generate_periodic_windows(start, split, months):
        try:
            print(f"🔹 National {ws}→{we}")
            df = fetch_data(ind_nat, ws, we, token, base_url)
            df = df[df["geo_id"] == geo_nat]
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Erreur nat {ws}-{we}: {e}")
        time.sleep(random.uniform(1, 3))

    if not dfs:
        return pd.DataFrame(columns=["datetime", "geo_id", "value"])

    df_nat = pd.concat(dfs).drop_duplicates("datetime").sort_values("datetime")
    save_hourly(df_nat, raw_dir / "España_pvpc_hourly.csv")
    return df_nat

def duplicate_to_regions(df_nat: pd.DataFrame, cfg, raw_dir):
    """
    Duplique la série nationale dans chaque fichier régional avant 2021-06.
    """
    for geo_id, name in cfg["pvpc"]["regional"]["geo_ids"].items():
        # On ne conserve que datetime+value, pas geo_id
        df_copy = df_nat[["datetime", "value"]].copy()
        save_hourly(df_copy, raw_dir / f"{name}_pvpc_hourly.csv")
        print(f"🌀 Duplication → {name}_pvpc_hourly.csv")

def process_regional(cfg, split, end, raw_dir, months=3):
    """
    Récupère et complète les données régionales (2021-06 → 2025-05).
    """
    token   = cfg["esios"]["token"]
    base_url= cfg["esios"]["base_url"]
    ind_reg = cfg["pvpc"]["regional"]["indicator_id"]
    geo_map = cfg["pvpc"]["regional"]["geo_ids"]

    dfs = []
    for ws, we in generate_periodic_windows(split, end, months):
        try:
            print(f"🔹 Regional {ws}→{we}")
            df = fetch_data(ind_reg, ws, we, token, base_url)
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Erreur reg {ws}-{we}: {e}")
        time.sleep(random.uniform(1, 3))

    if not dfs:
        return

    df_all = pd.concat(dfs).drop_duplicates(["datetime","geo_id"]).sort_values(["geo_id","datetime"])
    # Sauvegarde par région, en append et sans geo_id dans le CSV
    for geo_id, name in geo_map.items():
        df_region = df_all[df_all["geo_id"] == geo_id][["datetime","value"]]
        save_hourly(df_region, raw_dir / f"{name}_pvpc_hourly.csv")
        print(f"✅ {name}_pvpc_hourly.csv complété.")

    # Mise à jour nationale : moyenne post-2021
    df_avg = df_all.groupby("datetime")["value"].mean().reset_index()
    save_hourly(df_avg, raw_dir / "España_pvpc_hourly.csv")
    print("✅ National moyenne post-2021 mise à jour.")

def main_pvpc(start_date=None, end_date=None, config_path="config.yaml", months=3):
    """
    Main : national avant split, duplication, puis régional après split.
    """
    cfg     = load_config(config_path)
    dates   = cfg["dates"]
    split   = "2021-06-01"
    start   = start_date or dates["start_date"]
    end     = end_date   or dates["end_date"]
    raw_dir = Path(cfg["paths"]["data"]["external"]) / "prices"
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_nat = process_national(cfg, start, split, raw_dir, months)
    if not df_nat.empty:
        duplicate_to_regions(df_nat, cfg, raw_dir)
    process_regional(cfg, split, end, raw_dir, months)


# weather_data

def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def generate_windows(start_date: str, end_date: str, months: int = 1):
    """Découpe start–end en intervalles de `months` mois."""
    windows, current = [], datetime.fromisoformat(start_date)
    final = datetime.fromisoformat(end_date)
    while current < final:
        nxt = current + relativedelta(months=months) - timedelta(days=1)
        if nxt > final: nxt = final
        windows.append((current.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        current = nxt + timedelta(days=1)
    return windows

def fetch_open_meteo(lat, lon, start, end, vars):
    """Appelle l’API Open-Meteo pour un sous-intervalle donné."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(vars)
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json().get("hourly", {})
    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("datetime").drop(columns="time")

def save_csv(df, path, index_label="datetime"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label=index_label)

def main_weather(config_path="config.yaml", window_months=1):
    cfg    = load_config(config_path)
    coords = cfg["external"]["zones_coords"]
    start  = cfg["dates"]["start_date"]
    end    = cfg["dates"]["end_date"]
    vars   = ["temperature_2m","relative_humidity_2m","wind_speed_10m",
              "cloudcover","shortwave_radiation","precipitation"]
    out    = Path(cfg["paths"]["data"]["external"]) / "weather"

    # pour chaque zone, on découpe en fenêtres
    for zone, c in coords.items():
        print(f"→ {zone}")
        dfs = []
        for ws, we in generate_windows(start, end, months=window_months):
            try:
                print(f"  • {ws} → {we}")
                dfw = fetch_open_meteo(c["lat"], c["lon"], ws, we, vars)
                dfs.append(dfw)
            except requests.HTTPError as e:
                print(f"    ⚠️ Erreur {ws}-{we}: {e}")
            time.sleep(random.uniform(0.5, 3))
        if not dfs:
            print(f"  ❌ Pas de données pour {zone}")
            continue

        # Concaténation et dé-duplication
        df_all = pd.concat(dfs).sort_index().drop_duplicates()
        # Sauvegarde horaire
        save_csv(df_all, out / f"{zone}_weather_hourly.csv")
        # Sauvegarde journalière
        df_all.resample("D").mean(numeric_only=True) \
              .to_csv(out / f"{zone}_weather_daily.csv", index_label="date")
        print(f"  ✅ Météo sauvegardée pour {zone}")


# holidays_data

def filter_holidays(hol_obj, start: str, end: str) -> pd.DataFrame:
    """
    Extrait les jours fériés de hol_obj (HolidayBase) entre start et end.
    Retourne un DataFrame colonnes ['date','holiday'].
    """
    records = []
    for date, name in hol_obj.items():
        ds = date.strftime("%Y-%m-%d")
        if start <= ds <= end:
            records.append({"date": pd.to_datetime(date), "holiday": name})
    if not records:
        return pd.DataFrame(columns=["date", "holiday"])
    return pd.DataFrame(records).drop_duplicates().sort_values("date")

def main_holidays(config_path: str = "config.yaml"):
    cfg     = load_config(config_path)
    start   = cfg["dates"]["start_date"]
    end     = cfg["dates"]["end_date"]
    years   = list(range(int(start[:4]), int(end[:4]) + 1))
    country = cfg["external"]["holidays_country"]
    # Subdivisions pour les régions
    subs_map = {
        "Peninsule_Iberique": "AN",
        "Baleares":           "IB",
        "Canarias":           "CN",
        "Ceuta":              "CE",
        "Melilla":            "ML"
    }
    outdir = Path(cfg["paths"]["data"]["external"]) / "holidays"
    outdir.mkdir(parents=True, exist_ok=True)

    # Fichier national
    hol_nat = holidays.Spain(years=years)  # classe Spain importe holidays
    df_nat  = filter_holidays(hol_nat, start, end)
    df_nat.to_csv(outdir / "spain_holidays.csv", index=False)
    print("✅ spain_holidays.csv généré")

    # Fichiers régionaux
    for region, subdiv in subs_map.items():
        # on précise l'année et la subdivision
        hol_reg = holidays.Spain(years=years, subdiv=subdiv)
        df_reg  = filter_holidays(hol_reg, start, end)
        filename = f"{region}_holidays.csv"
        df_reg.to_csv(outdir / filename, index=False)
        print(f"✅ {filename} généré")








































# # weather_data

# def load_config(config_path: str = "config.yaml") -> dict:
#     """
#     Lit le fichier YAML de configuration et remplace ${VARNAME} par les variables d'environnement.
#     """
#     import yaml
#     with open(config_path, "r") as f:
#         cfg = yaml.safe_load(f)
#     for sec in cfg.values():
#         if isinstance(sec, dict):
#             for k, v in sec.items():
#                 if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
#                     sec[k] = os.getenv(v[2:-1])
#     return cfg

# def fetch_weather_hourly(api_key: str,
#                          lat: float,
#                          lon: float,
#                          start: str,
#                          end: str) -> pd.DataFrame:
#     """
#     Récupère les données horaires via l'API One Call Historical d'OpenWeatherMap.
#     - start/end en 'YYYY-MM-DD'
#     Retourne un DataFrame avec colonnes ['datetime','temp','humidity','wind_speed',...]
#     """
#     url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
#     dfs = []
#     dt = datetime.fromisoformat(start)
#     last = datetime.fromisoformat(end)
#     while dt <= last:
#        # Exemple de requête v3.0
#         params = {
#             "lat": 40.4168,
#             "lon": -3.7038,
#             "dt": 1430478000,   # timestamp UNIX
#             "appid": api_key,
#             "units": "metric"
#         }
#         resp = requests.get(url, params=params)
#         resp.raise_for_status() # Vérifie le statut de la réponse
#         data = resp.json().get("hourly", [])
#         if data:
#             df = pd.DataFrame(data)
#             df["datetime"] = pd.to_datetime(df["dt"], unit="s", utc=True)
#             dfs.append(df)
#         time.sleep(random.uniform(1, 2))  # pause pour ne pas surcharger l'API
#         dt += timedelta(days=1)
#     if not dfs:
#         return pd.DataFrame()
#     df_all = pd.concat(dfs).set_index("datetime").sort_index()
#     return df_all

# def save_weather(df: pd.DataFrame,
#                  zone_name: str,
#                  raw_dir: Path):
#     """
#     Sauvegarde les fichiers météo horaires et journaliers pour une zone donnée.
#     - df : DataFrame indexé par datetime
#     - raw_dir : répertoire racine data/external/weather/
#     """
#     hourly_dir = raw_dir / "weather"
#     hourly_dir.mkdir(parents=True, exist_ok=True)

#     # Horaire
#     hourly_path = hourly_dir / f"{zone_name}_weather_hourly.csv"
#     df.to_csv(hourly_path, index_label="datetime")

#     # Journalier : moyenne des variables
#     daily = df.resample("D").mean(numeric_only=True)
#     daily_path = hourly_dir / f"{zone_name}_weather_daily.csv"
#     daily.to_csv(daily_path, index_label="date")

# def main_weather(config_path: str = "config.yaml"):
#     """
#     Point d'entrée pour récupérer et sauvegarder la météo pour toutes les zones.
#     """
#     cfg = load_config(config_path)
#     zones = cfg["external"]["zones_coords"]
#     start = cfg["dates"]["start_date"]
#     end   = cfg["dates"]["end_date"]
#     api_key = cfg["external"]["openweather_key"]
#     raw_dir = Path(cfg["paths"]["data"]["external"])
    
#     for zone_name, coord in zones.items():
#         lat, lon = coord["lat"], coord["lon"]
#         print(f"🌤️ Récupération météo pour {zone_name} ({start}→{end})")
#         df_h = fetch_weather_hourly(api_key, lat, lon, start, end)
#         if df_h.empty:
#             print(f"  ⚠️ Aucune donnée pour {zone_name}")
#             continue
#         save_weather(df_h, zone_name, raw_dir)
#         print(f"  ✅ Fichiers météo sauvegardés pour {zone_name}")
