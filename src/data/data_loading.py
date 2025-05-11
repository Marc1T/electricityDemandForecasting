# src/data/data_loading.py

import os
import requests
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import random


def to_esios_date(iso_date: str) -> str:
    """
    Convertit une date ISO 'YYYY-MM-DD' en format 'DD-MM-YYYYT%H' attendu par l'API ESIOS.
    """
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    return dt.strftime("%d-%m-%YT%H")

def generate_periodic_windows(start_date: str,
                              end_date: str,
                              months: int = 1
                             ) -> list[tuple[str,str]]:
    """
    Génère des sous-périodes de 'months' mois entre start_date et end_date.
    Dates en entrée/sortie au format 'YYYY-MM-DD'.
    """
    windows = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end     = datetime.strptime(end_date,   "%Y-%m-%d")
    
    while current_start < final_end:
        # fin de fenêtre = début + months - 1 jour
        next_end = current_start + relativedelta(months=months) - timedelta(days=1)
        if next_end > final_end:
            next_end = final_end
        windows.append((
            current_start.strftime("%Y-%m-%d"),
            next_end.strftime("%Y-%m-%d")
        ))
        # on repart au jour suivant
        current_start = next_end + timedelta(days=1)
    
    return windows

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Lit le fichier YAML de configuration et renvoie un dict.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    # Remplace ${VARNAME} par la vraie variable d'env si nécessaire
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    var = v[2:-1]
                    section[k] = os.getenv(var)
    return cfg

def fetch_indicator_hourly(indicator_id: int,
                           token: str,
                           start_date: str,
                           end_date: str,
                           base_url: str) -> pd.DataFrame:
    """
    Interroge l'API ESIOS pour un indicateur entre start_date et end_date.
    Renvoie un DataFrame avec index horodaté et une colonne 'value'.
    """
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "x-api-key": f"{token}"
    }
    params = {
        "start_date": start_date,
        "end_date": end_date
    }
    url = f"{base_url}/{indicator_id}"
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()["indicator"]["values"]
    
    # Construction du DataFrame
    df = pd.DataFrame(data)

    df = df.groupby("datetime").agg({"value": "sum"}).reset_index()

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")[["value"]]
    return df

def save_hourly_and_daily(df_hourly: pd.DataFrame,
                          zone_name: str,
                          raw_dir: Path,
                          freq: str = "D") -> None:
    """
    Sauvegarde le DataFrame horaire et son agrégation journalière.
    Génère deux fichiers CSV dans raw_dir :
      - <zone_name>_hourly.csv
      - <zone_name>_daily.csv
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Fichier horaire
    hourly_path = raw_dir / f"{zone_name}_hourly.csv"
    df_hourly.to_csv(hourly_path, index_label="datetime")
    
    # Agrégation journalière
    df_daily = df_hourly.resample(freq).sum()
    daily_path = raw_dir / f"{zone_name}_daily.csv"
    df_daily.to_csv(daily_path, index_label="date")

def main(start_date: str = None,
         end_date:   str = None,
         config_path:str = "config.yaml",
         indicator_knowms: pd.DataFrame = None,
         window_months: int = 1):
    # --- Chargement config ---
    cfg        = load_config(config_path)
    token      = cfg["esios"]["token"]
    base_url   = cfg["esios"]["base_url"]
    paths      = cfg["paths"]["data"]
    dates_cfg  = cfg["dates"]
    
    if indicator_knowms is not None:
        # Filtre les indicateurs connus
        indicators = indicator_knowms
    else:
        indicators = cfg["indicators"]

    # --- Dates globales ---
    start = start_date or dates_cfg["start_date"]
    end   = end_date   or dates_cfg["end_date"]
    
    raw_dir = Path(paths["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for ind_id, zone in indicators.items():
        print(f"\n▶️  Zone: {zone} (ID={ind_id})")
        
        # Génération de fenêtres de window_months mois
        windows = generate_periodic_windows(start, end, months=window_months)
        dfs = []
        
        for ws, we in windows:
            ws_esios = to_esios_date(ws)
            we_esios = to_esios_date(we)
            print(f"  └─ requête {ws} → {we}")
            
            try:
                df_part = fetch_indicator_hourly(
                    indicator_id=ind_id,
                    token=token,
                    start_date=ws_esios,
                    end_date=we_esios,
                    base_url=base_url
                )
                dfs.append(df_part)
            except requests.exceptions.HTTPError as e:
                print(f"    ⚠️ Erreur sur {ws}-{we} : {e}")
                # ici vous pouvez décider de retry ou skip
        
        # Concaténation et dé-duplication éventuelle
        if dfs:
            df_hourly = pd.concat(dfs).sort_index()
            df_hourly = df_hourly[~df_hourly.index.duplicated(keep='first')]
            
            # Sauvegarde horaires + journaliers
            save_hourly_and_daily(df_hourly, zone, raw_dir)
            print(f"   ▶️  Fichiers écrits: {zone}_hourly.csv & {zone}_daily.csv")
        else:
            print(f"    ❌ Aucune donnée récupérée pour {zone}")
        # Pause entre 1 et 5 secondes
        time.sleep(random.uniform(1, 5))


# if __name__ == "__main__":
#     # Exécution directe possible :
#     #   python src/data/data_loading.py
#     main()
