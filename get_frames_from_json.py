import os
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd

import json


video_root = "D:\\soccernetDataset\\"

def save_dicts_to_txt(dict_list, filename="output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, item in enumerate(dict_list, start=1):
            f.write(f"Item {i}:\n")
            for key, value in item.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")  # Add a newline between items
    print(f"Saved {len(dict_list)} items to '{filename}'")


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
    print('json info not found')
    return ''  # If not found


def parse_filename(filename):
    # Remove prefix and suffix
    if filename.startswith("output_"):
        filename = filename[len("output_"):]
    if filename.endswith("_Labels-caption.txt"):
        filename = filename[:-len("_Labels-caption.txt")]

    parts = filename.split("_")
    # League is first two parts (e.g., spain + laliga)
    league = f"{parts[0]}_{parts[1]}"
    # Year is third part
    year = parts[2]
    # Match = the rest
    match = "_".join(parts[3:])

    return league, year, match


def construct_dict_play(line, id, goal_bool, end_time_offset):
    new = line.split(maxsplit = 3)
    text_description = new[-1]
    match_half_time = new[0]
    play_timestamp = new[2]
    play_min, play_sec = map(int, play_timestamp.split(":"))
    play_start = play_min*60 + play_sec

    goal_play = {'id': id,
                    'play_is_goal': goal_bool,
                    'match_ht': match_half_time,
                    'play_start': play_start,
                    'play_end': play_start+end_time_offset,
                     'commentary': text_description,}
    return goal_play



folder_data= "data\goal_files"
list_dir = os.listdir(folder_data)

all_goals =[]
all_non_goals= []
other_plays = []
all_plays = []


#process txt with format: 
#       'Goal 1 - 13:36 commentary'.
#and filename: 
#       output_england_epl_2014-2015_2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal_Labels-caption

for filename in list_dir:
    league, year, match = parse_filename(filename)
    end_time_offset = 10
    start_time_offset = 0
    #read file with goals and no goals
    complete_fn = folder_data + "\\" + filename
    with open(complete_fn, mode ="r") as file:
        for line in file:
            splitted =line.split(maxsplit = 1)
            print(splitted[0])
            if splitted[0] == "Goal":
                goal_bool = True
            elif splitted[0] == "No":
                goal_bool = False
                splitted = splitted[1].split(maxsplit = 1)

            play = construct_dict_play(splitted[1], len(all_plays), goal_bool, end_time_offset)

            if int(play['match_ht']) > 2:
                other_plays.append(play)
                continue

            cleaned_match = match.replace("_", " ") #soccernet dataset folders dont use _ 
            video_path = f"{video_root}{league}\\{year}\\{cleaned_match}\\{play['match_ht']}_"
            play['league'] = league
            play['year'] = year
            play['match'] = cleaned_match
            play['video_path'] = video_path
            

            if goal_bool:
                all_goals.append(play)
            else:
                all_non_goals.append(play)

            all_plays.append(play)
                

# ADD MatchTime times, when they exist..
for play in all_plays:
    root_path = 'D:\\MatchTime'
    json_path = f"{root_path}\\{play['league']}_{play['year']}\\{play['match']}\\Labels-caption.json" 
    play_timestamp = get_contrastive_aligned_time(json_path, play['commentary'])
    if play_timestamp!='':
        parts = play_timestamp.split("-")
        play_timestamp = parts[1]
        play_min, play_sec = map(int, play_timestamp.split(":"))
        play_start = play_min*60 + play_sec
        play['play_start_MT'] = play_start
        play['play_end_MT']= play_start+end_time_offset
    else:
        play['play_start_MT'] = 0
        play['play_end_MT']= 0


os.makedirs('results', exist_ok=True)

print("Saving to txt...")
save_dicts_to_txt(all_goals, "results\\all_goals.txt")
save_dicts_to_txt(all_non_goals, "results\\all_non_goals.txt")
save_dicts_to_txt(other_plays, "results\\other_plays.txt")

new_order = ['id', 'play_is_goal', 'league', 'year', 'match', 'video_path',  'match_ht', 'commentary', 'play_start', 'play_end', 'play_start_MT','play_end_MT' ]
print("Saving to csv...")
if all_goals:
    all_goals_df = pd.DataFrame(all_goals)
    all_goals_df = all_goals_df[new_order]
    all_goals_df.to_csv("results/goals_MT.csv", index=False)

if all_non_goals:
    all_non_goals_df = pd.DataFrame(all_non_goals)
    all_non_goals_df = all_non_goals_df[new_order]
    all_non_goals_df.to_csv("results/non_goals_MT.csv", index=False)

if all_plays:
    all_plays = pd.DataFrame(all_plays)
    all_plays = all_plays[new_order]
    all_plays.to_csv("results/all_plays_MT.csv", index=False)

print("finished")

            

