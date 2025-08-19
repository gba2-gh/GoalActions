import os
import pandas as pd
import json

#This file
# ADDs MatchTime times, when they exist..
#Changes 'label' from str to boolean, excludes penalties
#adds id value
#creates updated json and csv file

def normalize(text):
    return text.strip().lower()

def get_contrastive_aligned_time(json_path,  target_description):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations = data.get("annotations", [])

        for entry in annotations:
            if (normalize(entry.get("description", "")) == normalize(target_description)):
                return entry.get("contrastive_aligned_gameTime")
    print(f'{json_path}: json info not found')
    return ''  # If not found


with open('results/extract_actions_soccernet/all_events.json', mode='r', encoding='utf-8') as f:
    all_plays = json.load(f)


for play in all_plays:
    root_path = 'D:\\MatchTime'
    json_path = f"{root_path}\\{play['league']}_{play['season']}\\{play['match']}\\Labels-caption.json" 
    play_timestamp = get_contrastive_aligned_time(json_path, play['caption'])
    if play_timestamp!='':
        parts = play_timestamp.split("-")
        play_timestamp = parts[1]
        play_min, play_sec = map(int, play_timestamp.split(":"))
        play_start = play_min*60 + play_sec
        play['time_MT'] = play_start
    else:
        play['time_MT'] = play['time']

i=0
for play in all_plays:
    play['id'] = i
    i+=1
    if play['label'] == 'Goal':
        play['label'] = True
    elif play['label'] == 'No Goal':
        play['label'] = False

with open('results/all_plays.json', mode='w', encoding='utf-8') as f:
    json.dump(all_plays, f, indent=2)

new_order = ['id', 'hash', 'label', 'league', 'season', 'match', 'half_time', 'caption', 'time', 'time_MT']
print("Saving to csv...")

all_plays_df = pd.DataFrame(all_plays)
all_plays_df = all_plays_df[new_order]
all_plays_df.to_csv("results/all_plays.csv", index=False)