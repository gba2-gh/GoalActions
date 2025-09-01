import os
import pandas as pd
import json
from create_videos_from_csv import ClipExtractor
from football_action_extraction import FootballActionExtractor
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
            play['id'] = int(i)
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

    #all_plays = all_plays.to_dict(orient="records")

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


def save_results(all_plays_updated):
    #Save results
    print("Saving to json...")
    with open('results/all_plays_updated.json', mode='w', encoding='utf-8') as f:
        json.dump(all_plays_updated, f, indent=2)

    all_plays_updated_df = pd.DataFrame(all_plays_updated)
    #new_order = ['id', 'hash', 'label', 'league', 'season', 'match', 'half_time', 'caption', 'score', 'narration', 'time']
    print("Saving to csv...")
    all_plays_df = pd.DataFrame(all_plays_updated_df)
    #all_plays_df = all_plays_df[new_order]
    all_plays_df.to_csv("results/all_plays_updated.csv", index=False)



from sklearn.model_selection import train_test_split

def divide_data(df):

    # Stratified split by label (80/20)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42  # fixed seed for reproducibility
    )

    train_df.to_csv("results/divided_data/train.csv", index=False)
    val_df.to_csv("results/divided_data/val.csv", index=False)


    # Save back to JSON if needed
    train_data = train_df.to_dict(orient="records")
    test_data = val_df.to_dict(orient="records")

    with open("results/divided_data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open("results/divided_data/val.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    print("Split completed and saved to train.json and test.json")




def process_all_events_updated():
    all_plays_df = pd.read_csv('results/all_plays_sorted.csv')

    ##update specific problems with updated file
    all_plays_df['time'] = all_plays_df['end_time']
    all_plays_df['label'] = all_plays_df['label'].apply(lambda x: 1 if x == 'TRUE' else 0)

        #Get not found
    all_plays_nf = all_plays_df[all_plays_df['time_label'] == 'NF']
    all_plays_nf.to_csv("results/nf_plays.csv", index=False)

    not_updated_df  = all_plays_df[~all_plays_df['updated']]
    all_plays_nf.to_csv("results/not_updated_plays.csv", index=False)


        #get only updated = TRUE
    all_plays_updated_df  = all_plays_df[all_plays_df['updated']]

    all_plays_updated = all_plays_updated_df.to_dict(orient="records")

    # convert to dict and add commentary
    add_asr_commentary(all_plays_updated, start_offset=10, end_offset=10)

    #Find extract action vector
    extractor = FootballActionExtractor()
    extractor.extract_from_dict(all_plays_updated)

    save_results(all_plays_updated)
    
    all_plays_updated_df = pd.DataFrame(all_plays_updated)
    divide_data(all_plays_updated_df)

    #Extact clips and save
    clip_extractor = ClipExtractor()
    clip_extractor.by_end =True # consider 'Time' as an end_time and not start_time
    video_output_folder = "results\\video_goal_output_60_updated"
    clip_extractor.extract_clips(all_plays_updated_df, video_output_folder) # doesnt need to be updated


if __name__ == "__main__":
    
    #process_all_events()

    process_all_events_updated()


    


