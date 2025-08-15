# for sn-echoes dataset
import os
import json
import re

# === CONFIGURATION ===
LABEL_DIR = r"E:\FrankGuo\SoccerNet\goal_results_2023"
ASR_BASE_DIR = r"E:\FrankGuo\sn-echoes\Dataset\whisper_v1_en"
OUTPUT_DIR = r"E:\FrankGuo\SoccerNet\crossMatch"

# === DEBUG SETTINGS ===
DEBUG = True  # Set to False to reduce debug output
DEBUG_VERBOSE = False  # Set to False to show only critical debug info

# === HELPERS ===
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def verbose_debug_print(*args, **kwargs):
    if DEBUG and DEBUG_VERBOSE:
        print(*args, **kwargs)

def parse_time(half: int, mmss: str) -> float:
    minutes, seconds = map(int, mmss.strip().split(":"))
    total_seconds = minutes * 60 + seconds
    debug_print(f"🕒 Parsed time — Half: {half}, Timestamp: {mmss} → {total_seconds:.2f} seconds")
    return total_seconds

def load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        debug_print(f"❌ Error loading JSON file {path}: {str(e)}")
        return None

def find_caption_window(asr_data, target_time, delta=6.0):
    debug_print(f"🔍 Searching captions around {target_time:.2f}s ± {delta}s...")
    result = []
    window_start = target_time - delta
    window_end = target_time+5
    
    # Access the segments dictionary from the ASR data
    segments = asr_data.get("segments", {})
    
    for key, value in segments.items():
        if isinstance(value, list) and len(value) == 3:
            start, end, text = value
            in_window = (start <= window_end) and (end >= window_start)

            if DEBUG_VERBOSE:
                debug_line = (
                    f"🔸 ID {key} | Start: {start:.2f}, End: {end:.2f} | "
                    f"Window: {window_start:.2f}–{window_end:.2f} | "
                    f"{'✅ MATCH' if in_window else '❌'} → \"{text[:50]}\""
                )
                verbose_debug_print(debug_line)

            if in_window:
                result.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

    if not result:
        debug_print("⚠️ No captions found in this time window.")
    debug_print(f"✅ Total matches: {len(result)}\n")
    return result

def filename_to_match_dir(filename):
    verbose_debug_print(f"DEBUG: filename = {filename}")
    pattern = r"output_([a-zA-Z0-9_]+)_(\d{4}-\d{4})_(\d{4}-\d{2}-\d{2})_-_(\d{2}-\d{2})_(.*)_Labels-caption.txt"
    match = re.match(pattern, filename)
    if not match:
        debug_print("DEBUG: regex did not match.")
        return None
    competition, season, date, time, rest = match.groups()
    verbose_debug_print(f"DEBUG: competition={competition}, season={season}, date={date}, time={time}, rest={rest}")
    match_dir_name = f"{date} - {time} {rest.replace('_', ' ')}"
    full_path = os.path.join(ASR_BASE_DIR, competition, season, match_dir_name)
    verbose_debug_print(f"DEBUG: constructed path = {full_path}")
    return full_path

# === MAIN PROCESSING ===
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

        json_paths = {
            1: os.path.join(match_dir, "1_asr.json"),
            2: os.path.join(match_dir, "2_asr.json")
        }

        # Load ASR data with debug info
        asr_data = {}
        for half in [1, 2]:
            if os.path.exists(json_paths[half]):
                data = load_json(json_paths[half])
                if data is None:
                    asr_data[half] = {"segments": {}}
                    continue
                    
                asr_data[half] = data
                
                # Debug: Show time range of this half
                segments = data.get("segments", {})
                times = [
                    (v[0], v[1]) for v in segments.values()
                    if isinstance(v, list) and len(v) == 3
                ]
                if times:
                    start = min(t[0] for t in times)
                    end = max(t[1] for t in times)
                    debug_print(f"⏱ ASR Half {half}: {len(times)} captions | Range: {start:.2f}s → {end:.2f}s")
                else:
                    debug_print(f"⚠️ No valid captions in Half {half}")
            else:
                debug_print(f"⚠️ Missing ASR JSON for half {half}: {json_paths[half]}")
                asr_data[half] = {"segments": {}}

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

            matches = find_caption_window(asr_data.get(half, {}), timestamp, delta=45.0)

            results.append({
                "event_type": event_type,
                "half": half,
                "timestamp": mmss,
                "seconds": timestamp,
                "captions": matches
            })

        # Save result
        output_path = os.path.join(OUTPUT_DIR, file.replace("_Labels-caption.txt", "_captions.json"))
        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            debug_print(f"✅ Saved matched captions: {output_path}\n{'='*80}")
        except Exception as e:
            debug_print(f"❌ Error saving results to {output_path}: {str(e)}")

if __name__ == "__main__":
    process_labels()