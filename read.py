# mqtt_llm_bridge.py
import json
import logging
import re
import threading
import paho.mqtt.client as mqtt
from ollama import chat
from ollama._types import ResponseError

BROKER = "test.mosquitto.org"
TOPIC_IN = "/dforirdod/PKM/battle/info"
TOPIC_OUT = "/dforirdod/PKM/battle/move"
MODEL = "gemma3"  # or another installed model tag, e.g. "gemma2:2b-instruct"

SYSTEM = """You are a Pokémon Red (Gen 1) battle planner. 
You will receive exactly one JSON object called "scene" describing the current battle state.
Your job: pick the best move index for the player's active Pokémon.

Rules:
- Return ONLY a compact JSON object: {"move_nb": i, "move_name": "<name>", "reason": "<why>"} where i ∈ {1,2,3,4}. No prose, no extra keys.
- Consider Gen 1 logic: move power/accuracy, type effectiveness, status impact, PP, both HP/levels, and KO risk.
- NEVER choose a move with PP ≤ 0, undefined/bugged ("NA") or accuracy < 0.
- With very low player HP, prioritize survival (sleep/paralysis/attack-down) over raw damage if it reduces KO risk.
- Prefer reliable control (sleep/paralysis) when it creates a safe setup.
- If several moves are similar, prefer higher accuracy and PP conservation.
- If information is missing, make the safest reasonable assumption.

Output format:
- Exactly: {"move_nb": i, "move_name": "<name>", "reason": "<why>"}
- No markdown, no comments, no trailing text.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pkm-mqtt")

# Memory of last battle_turn per battle_id
_last_turn_by_battle = {}
_state_lock = threading.Lock()

def _extract_scene(payload: dict):
    """
    Supports both formats:
      1) {"scene": {..., "battle_turn": X, ...}, "battle_id": ..., ...}
      2) {..., "scene": {...}, "battle_id": ..., "turn": ...}
    Returns: (scene_obj, scene_core, battle_id, battle_turn)
    """
    root = payload
    if "scene" in root and isinstance(root["scene"], dict):
        scene_obj = root
        scene_core = root["scene"]
    else:
        # handle case where "scene" is directly published
        scene_obj = {"scene": root}
        scene_core = root

    battle_id = scene_obj.get("battle_id")
    battle_turn = None
    if isinstance(scene_core, dict) and "battle_turn" in scene_core:
        battle_turn = scene_core["battle_turn"]
    else:
        battle_turn = scene_obj.get("turn")

    return scene_obj, scene_core, battle_id, battle_turn

def _safe_json_extract(s: str) -> dict:
    """Try strict JSON first; if not, extract the first {...} block."""
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty model output")
    # First try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # Fallback: extract the first balanced JSON object
    m = re.search(r'\{.*\}', s, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in: {s[:120]!r}...")
    return json.loads(m.group(0))

def _validate_move(scene_obj: dict, data: dict) -> tuple[int, str, str]:
    if not all(k in data for k in ("move_nb", "move_name", "reason")):
        raise ValueError(f"Missing keys in {data}")
    i = int(data["move_nb"])
    if i not in (1,2,3,4):
        raise ValueError(f"Illegal move index: {i}")
    mv = scene_obj["scene"]["playerPKM"]["moves"][i-1]
    # Guardrails
    if mv["name"].upper() == "NA": raise ValueError("Picked NA")
    if mv["pp"][0] <= 0: raise ValueError("No PP")
    if mv["accuracy"] is not None and mv["accuracy"] < 0: raise ValueError("Negative accuracy")
    return i, mv["name"], data["reason"]

def _fallback_rule(scene_obj: dict) -> dict:
    """Tiny deterministic fallback: prefer sleep, else highest power non-NA with PP>0."""
    moves = scene_obj["scene"]["playerPKM"]["moves"]
    # 1) sleep if available & PP>0
    for idx, mv in enumerate(moves, start=1):
        if mv["name"].upper() == "SING" and mv["pp"][0] > 0 and (mv.get("accuracy", 0) >= 0):
            return {"move_nb": idx, "move_name": mv["name"], "reason": "Fallback: sleep for survival"}
    # 2) max (power * accuracy%) heuristic
    best = None
    for idx, mv in enumerate(moves, start=1):
        if mv["name"].upper() == "NA" or mv["pp"][0] <= 0 or mv.get("accuracy", -1) < 0:
            continue
        acc = mv.get("accuracy", 100.0) or 100.0
        score = (mv.get("power", 0) or 0) * (acc / 100.0)
        if best is None or score > best[0]:
            best = (score, idx, mv["name"])
    if best:
        return {"move_nb": best[1], "move_name": best[2], "reason": "Fallback: highest power×accuracy"}
    # 3) last resort
    return {"move_nb": 1, "move_name": moves[0]["name"], "reason": "Fallback: default slot 1"}

def decide_move(scene_obj: dict) -> str:
    payload = json.dumps({"scene": scene_obj})
    # --- First attempt: force JSON output ---
    try:
        resp = chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": payload},
            ],
            options={
                "format": "json",       # <- forces JSON output in Ollama
                "temperature": 0.2,
                "num_predict": 48,
                "seed": 7,
                "stop": ["```", "\n\n"],
            },
        )
        raw = (resp.message.content or "").strip()
        data = _safe_json_extract(raw)
        i, name, reason = _validate_move(scene_obj, data)
        return json.dumps({"move_nb": i, "move_name": name, "reason": reason})
    except Exception as e_first:
        # --- Retry once, stricter instruction, no format lock (some models ignore 'format') ---
        retry_instr = {
            "role": "user",
            "content": json.dumps({
                "scene": scene_obj,
                "instruction": "Return ONLY JSON: {\"move_nb\": i, \"move_name\": \"<name>\", \"reason\": \"<why>\"} with i in {1,2,3,4}. No other text."
            })
        }
        try:
            resp = chat(
                model=MODEL,
                messages=[{"role":"system","content":SYSTEM}, retry_instr],
                options={"temperature":0.2, "num_predict":48, "seed":7, "stop":["```", "\n\n"]},
            )
            raw = (resp.message.content or "").strip()
            data = _safe_json_extract(raw)
            i, name, reason = _validate_move(scene_obj, data)
            return json.dumps({"move_nb": i, "move_name": name, "reason": reason})
        except Exception as e_second:
            # --- Final fallback: rule-based to keep the pipeline alive ---
            fb = _fallback_rule(scene_obj)
            return json.dumps(fb)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        logger.error(f"MQTT connect failed: {reason_code}")
        return
    logger.info("Connected to broker")
    client.subscribe(TOPIC_IN, qos=0)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception as e:
        logger.warning(f"Invalid JSON on {msg.topic}: {e}")
        return

    scene_obj, scene_core, battle_id, battle_turn = _extract_scene(payload)
    if battle_turn is None:
        logger.debug("No battle_turn found; ignoring")
        return

    key = battle_id if battle_id is not None else "_default"
    with _state_lock:
        last_turn = _last_turn_by_battle.get(key)

        if last_turn == battle_turn:
            # Same turn → skip
            logger.debug(f"Battle {key}: same turn {battle_turn}, ignoring")
            return

        # New turn detected → decide and publish
        try:
            raw = decide_move(scene_obj)
        except ResponseError as e:
            logger.error(f"Ollama error: {e}")
            return
        except Exception as e:
            logger.exception(f"Decision error: {e}")
            return

        client.publish(TOPIC_OUT, payload=raw, qos=2, retain=False)
        logger.info(f"Published move for battle {key} turn {battle_turn}: {raw}")

        _last_turn_by_battle[key] = battle_turn


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pkm-move-agent")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, port=1883, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
