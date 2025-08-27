import os
import pandas as pd
import json
from create_videos_from_csv import ClipExtractor
#This file
# ADDs MatchTime times, when they exist..
#Changes 'label' from str to boolean, excludes penalties
#adds id value
#creates updated json and csv file

def normalize(text):
    return text.strip().lower()

def clean_json_soccernet(all_plays):
    #remove non useful half times and change goal label to boolean
    i=0
    #accepted_plays = []
    for play in all_plays:
        if play['half_time'] >2:
            continue
        else:
            play['id'] = i
            i+=1
            if play['label'] == 'Goal':
                play['label'] = True
            elif play['label'] == 'No Goal':
                play['label'] = False

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

def add_matchtime_times(all_plays):
    ## add matchtime
    for play in all_plays:
        root_path = 'D:\\MatchTime'
        json_path = f"{root_path}\\{play['league']}_{play['season']}\\{play['match']}\\Labels-caption.json" 
        play_timestamp = get_contrastive_aligned_time(json_path, play['caption'])
        if play_timestamp!='':
            parts = play_timestamp.split("-")
            play_timestamp = parts[1]
            play_min, play_sec = map(int, play_timestamp.split(":"))
            play_start = play_min*60 + play_sec
            play['time'] = play_start
        

def add_asr_commentary(all_plays, start_offset = 30, end_offset = 30):
    ## add echoes 
    no_comm_goals=0
    no_comm_chances=0

    for play in all_plays:
        play_commentary =''
        root_path = 'D:\\SoccerNetEchoes\\sn-echoes\\Dataset\\complete'
        json_path = f"{root_path}\\{play['league']}\\{play['season']}\\{play['match']}\\{play['half_time']}_asr.json" 
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for key, value in data["segments"].items():
                    comment_start_time = value[0]
                    comment_end_time = value[1]
                    comment = value[2]
                    if play['time'] + end_offset < comment_start_time :
                        break

                    if play['time'] -start_offset < comment_start_time:
                        play_commentary = f'{play_commentary} [{comment_start_time}, {comment_end_time}], {comment};'
        
        if play_commentary == "":
            if play['label'] == True:
                no_comm_goals+=1
            else:
                no_comm_chances+=1

        play['narration'] = play_commentary

    print(f'no_comm_goals : {no_comm_goals}')         
    print(f'no_comm_chances : {no_comm_chances}') 



def process_all_events():

    with open('results/extract_actions_soccernet/all_events.json', mode='r', encoding='utf-8') as f:
        all_plays = json.load(f)

    clean_json_soccernet(all_plays)
    add_matchtime_times(all_plays)
    add_asr_commentary(all_plays)


    print("Saving to json...")
    with open('results/all_plays.json', mode='w', encoding='utf-8') as f:
        json.dump(all_plays, f, indent=2)

    #new_order = ['id', 'hash', 'label', 'league', 'season', 'match', 'half_time', 'caption', 'score', 'narration', 'time']
    print("Saving to csv...")

    all_plays_df = pd.DataFrame(all_plays)
    #ll_plays_df = all_plays_df[new_order]
    all_plays_df.to_csv("results/all_plays.csv", index=False)


def process_all_events_updated():
    all_plays_df = pd.read_csv('results/all_plays_sorted.csv')
    all_plays_df['time'] = all_plays_df['end_time']

    #Get not found
    all_plays_nf = all_plays_df[all_plays_df['time_label'] == 'NF']
    all_plays_nf.to_csv("results/all_plays_nf.csv", index=False)

    #get only updated
    updated_plays = all_plays_df[all_plays_df['updated']]
    all_plays_df = updated_plays

    all_plays = all_plays_df.to_dict(orient="records")
    #add_matchtime_times(all_plays)
    

    add_asr_commentary(all_plays, start_offset=10, end_offset=10)

    print("Saving to json...")
    with open('results/all_plays.json', mode='w', encoding='utf-8') as f:
        json.dump(all_plays, f, indent=2)

    #new_order = ['id', 'hash', 'label', 'league', 'season', 'match', 'half_time', 'caption', 'score', 'narration', 'time']
    print("Saving to csv...")
    all_plays_df = pd.DataFrame(all_plays)
    #all_plays_df = all_plays_df[new_order]
    all_plays_df.to_csv("results/all_plays_updated.csv", index=False)

    return all_plays_df


if __name__ == "__main__":
    clip_extractor = ClipExtractor()
    clip_extractor.by_end =True
    all_plays_df  = process_all_events_updated()
    video_output_folder = "results\\video_goal_output_60_updated"
    #clip_extractor.extract_clips(all_plays_df, video_output_folder)
