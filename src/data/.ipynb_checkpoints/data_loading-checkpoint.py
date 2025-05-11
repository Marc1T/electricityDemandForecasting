import os
import requests
import pandas as pd
from datetime import datetime
import yaml

# === Chargement de la configuration ===
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# === Téléchargement des données depuis l’API ESIOS ===
def download_demand_data(start_date, end_date, indicators=None, save_path=None):
    config = load_config()
    token = config["api"]["ree_token"]

    if indicators is None:
        indicators = {
            1293: "Peninsule_Iberique",
            10244: "Baleares",
            10243: "Canarias",
            1347: "Ceuta",
            1348: "Melilla",
            1345: "Mallorca_Menorca",
            1346: "Ibiza_Formentera",
            1340: "Lanzarote_Fuerteventura",
            1341: "Tenerife",
            1342: "La_Palma",
            1343: "La_Gomera",
            1344: "El_Hierro"
        }

    headers = {
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "Authorization": f"Token token={token}"
    }

    base_url = "https://api.esios.ree.es/indicators/{}"

    data_frames = []

    for indicator_id, region in indicators.items():
        print(f"🔄 Téléchargement pour {region} (ID {indicator_id})...")

        try:
            url = base_url.format(indicator_id)
            params = {
                "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            }

            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} pour {region}")
                continue

            values = response.json().get("indicator", {}).get("values", [])
            if not values:
                print(f"⚠️ Pas de données pour {region}")
                continue

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["region"] = region
            data_frames.append(df)

        except Exception as e:
            print(f"❌ Erreur pour {region}: {e}")

    if not data_frames:
        print("🚫 Aucun jeu de données récupéré.")
        return pd.DataFrame()

    full_df = pd.concat(data_frames).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        full_df.to_csv(save_path, index=False)
        print(f"✅ Données sauvegardées : {save_path}")

    return full_df
