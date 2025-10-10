import requests

class MemoryReader:
    """
    Reads game state data from the gosumemory JSON endpoint.
    """
    def __init__(self, url="http://127.0.0.1:24050/json"):
        self.url = url

    def get_game_state(self):
        """
        Fetches the latest game state including score, combo, accuracy, and hit counts.

        Returns:
            dict: A dictionary containing game state data.
                  Returns a default dictionary if data cannot be fetched.
        """
        try:
            resp = requests.get(self.url, timeout=0.1)
            data = resp.json()

            gameplay = data.get("gameplay", {})
            menu = data.get("menu", {})
            hits = gameplay.get("hits", {})

            state = {
                "game_state": menu.get("state", 0), # Default to 0 if not found
                "score": gameplay.get("score", 0),
                "combo": gameplay.get("combo", {}).get("current", 0),
                "accuracy": gameplay.get("accuracy", 100.0) / 100.0,
                "miss": hits.get("0", 0),
                "hit_50": hits.get("50", 0),
                "hit_100": hits.get("100", 0),
                "hit_300": hits.get("300", 0),
                "hit_geki": hits.get("geki", 0) # 'geki' is often used for perfect hits in mania
            }
            return state

        except (requests.RequestException, ValueError):
            # Return default values if osu! is not running, in menu, or data is malformed
            return {
                "game_state": 0, # Assuming 0 is a non-play state
                "score": 0,
                "combo": 0,
                "accuracy": 1.0,
                "miss": 0,
                "hit_50": 0,
                "hit_100": 0,
                "hit_300": 0,
                "hit_geki": 0
            }
