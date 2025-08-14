#cross match of the two kind of captions.
import os
import json
import re

LABEL_DIR = r"E:\FrankGuo\SoccerNet\goal_results_2023"
ASR_BASE_DIR = r"E:\FrankGuo\MatchTime\combined"
OUTPUT_DIR = r"E:\FrankGuo\MatchTime\crossMatch"

DEBUG = True
DEBUG_VERBOSE = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def verbose_debug_print(*args, **kwargs):
    if DEBUG and DEBUG_VERBOSE:
        print(*args, **kwargs)

def parse_time(half: int, mmss: str) -> float:
    minutes, seconds = map(int, mmss.strip().split(":"))
    return (half - 1) * 45 * 60 + minutes * 60 + seconds

def game_time_to_seconds(game_time_str: str) -> float:
    match = re.match(r"(\d+)\s*-\s*(\d{1,2}):(\d{2})", game_time_str)
    if not match:
        return None
    half, minutes, seconds = map(int, match.groups())
    return (half - 1) * 45 * 60 + minutes * 60 + seconds

def load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        debug_print(f"❌ Error loading JSON: {path} — {e}")
        return None

def find_caption_window_from_annotations(annotations, target_time, delta=45.0):
    result = []
    for ann in annotations:
        game_time_str = ann.get("event_aligned_gameTime") or ann.get("gameTime")
        if not game_time_str:
            continue

        ann_seconds = game_time_to_seconds(game_time_str)
        if ann_seconds is None:
            continue

        if abs(ann_seconds - target_time) <= delta:
            result.append({
                "gameTime": game_time_str,
                "seconds": ann_seconds,
                "description": ann.get("description", ""),
                "label": ann.get("label", ""),
                "important": ann.get("important", False),
                "position": ann.get("position", "")
            })

    debug_print(f"✅ Found {len(result)} captions near {target_time:.2f}s")
    return result

def filename_to_match_dir(filename):
    pattern = r"output_([a-zA-Z0-9_]+)_(\d{4}-\d{4})_(\d{4}-\d{2}-\d{2})_-_(\d{2}-\d{2})_(.*)_Labels-caption.txt"
    match = re.match(pattern, filename)
    if not match:
        return None
    competition, season, date, time, rest = match.groups()
    competition_season = f"{competition}_{season}"
    match_dir_name = f"{date} - {time} {rest.replace('_', ' ')}"
    return os.path.join(ASR_BASE_DIR, competition_season, match_dir_name)

def process_labels():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(LABEL_DIR):
        if not file.endswith("_Labels-caption.txt"):
            continue

        label_path = os.path.join(LABEL_DIR, file)
        match_dir = filename_to_match_dir(file)

        if not match_dir or not os.path.exists(match_dir):
            debug_print(f"❌ Match directory not found: {match_dir}")
            continue

        json_path = os.path.join(match_dir, "Labels-caption.json")
        if not os.path.exists(json_path):
            debug_print(f"⚠️ JSON file not found: {json_path}")
            continue

        data = load_json(json_path)
        if data is None:
            continue

        annotations = data.get("annotations", [])
        if not annotations:
            debug_print(f"⚠️ No annotations in: {json_path}")
            continue

        results = []

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            match = re.match(r"(Goal|No Goal)\s+(\d)\s*-\s*(\d{2}:\d{2})", line)
            if not match:
                continue

            event_type, half_str, mmss = match.groups()
            half = int(half_str)
            timestamp = parse_time(half, mmss)

            matches = find_caption_window_from_annotations(annotations, timestamp, delta=45.0)

            results.append({
                "event_type": event_type,
                "half": half,
                "timestamp": mmss,
                "seconds": timestamp,
                "captions": matches
            })

        output_path = os.path.join(OUTPUT_DIR, file.replace("_Labels-caption.txt", "_captions.json"))
        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            debug_print(f"✅ Saved: {output_path}\n{'='*80}")
        except Exception as e:
            debug_print(f"❌ Failed to save output: {e}")

if __name__ == "__main__":
    process_labels()
