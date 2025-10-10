import requests
import time

URL = "http://127.0.0.1:24050/json"

def get_osu_data():
    try:
        resp = requests.get(URL, timeout=0.5)
        data = resp.json()

        # --- Extract các phần cần thiết ---
        menu = data.get("menu", {})
        gameplay = data.get("gameplay", {})

        result = {
            # mods
            "mods": menu.get("mods"),
            #game mode
            "game_mode": menu.get("gameMode"),
            # score
            "score": gameplay.get("score"),
            # accuracy
            "accuracy": gameplay.get("accuracy"),
            # combo
            "combo_current": gameplay.get("combo", {}).get("current"),
            "combo_max": gameplay.get("combo", {}).get("max"),
            # hits
            "miss": gameplay.get("hits", {}).get("0"),
            "hit_50": gameplay.get("hits", {}).get("50"),
            "hit_100": gameplay.get("hits", {}).get("100"),
            "hit_300": gameplay.get("hits", {}).get("300"),
            "hit_geki": gameplay.get("hits", {}).get("geki"),
            "sliderBreaks": gameplay.get("hits", {}).get("sliderBreaks"),
        }

        return result

    except Exception as e:
        print("Error fetching data:", e)
        return None


# --- Loop realtime ---
while True:
    data = get_osu_data()
    time.sleep(0.2)  # poll mỗi 0.2s (~5Hz)