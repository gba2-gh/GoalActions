import os
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd

extraction_resolution= '224p'
exrtaction_format = 'mkv'
video_root = "D:/soccernetDataset"
clip_lenght = 5 #seconds

def extract_video_segment(video_path, start_time, end_time, output_path):
    if os.path.exists(output_path):
        print(f"Skipped: '{output_path}' already exists.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clip = VideoFileClip(video_path, audio=False)
    duration = clip.duration

    if start_time>= duration:
        print(f"Skipped: start_time ({start_time}s) >= video duration ({duration:.2f}s)")
        return

    # Clamp end_time to the video duration
    end_time = min(end_time, duration)
    clip = clip.subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264", audio=False)  # Output is .mp4 by default


def extract_clips(df, output_folder):
    for index,play in df.iterrows():
        if play['half_time'] == 3 or play['half_time'] == 4:
            continue
        #if index > 10:
           # break
        input_path = f"{video_root}/{play['league']}/{play['season']}/{play['match']}/{play['half_time']}_{extraction_resolution}.{exrtaction_format}"
        output_path = f"{output_folder}\\action_{play['id']}\\clip.mp4" #folder named action to keep consistency with XVARS
        print(f'processing output: {output_path}')
        extract_video_segment( input_path, play['time_MT'], play['time_MT'] + clip_lenght, output_path)
    return

def extract_extra_clips(df, output_folder, extra_lenght = 60):
    for index,play in df.iterrows():
        if play['half_time'] == 3 or play['half_time'] == 4:
            continue
        #if index > 10:
           # break
        input_path = f"{video_root}/{play['league']}/{play['season']}/{play['match']}/{play['half_time']}_{extraction_resolution}.{exrtaction_format}"
        #extra before
        output_path = f"{output_folder}\\action_{play['id']}\\clip_before.mp4" #folder named action to keep consistency with XVARS
        print(f'processing output: {output_path}')
        extract_video_segment( input_path, play['time_MT'] - extra_lenght, play['time_MT'] + clip_lenght, output_path)
        #extra after
        output_path = f"{output_folder}\\action_{play['id']}\\clip_after.mp4" #folder named action to keep consistency with XVARS
        print(f'processing output: {output_path}')
        extract_video_segment( input_path, play['time_MT'] , play['time_MT'] + clip_lenght + extra_lenght, output_path)

    return

output_folder = "results\\video_goal_output"
all_plays_df = pd.read_csv('results/all_plays.csv')
extract_clips(all_plays_df, output_folder)
extract_extra_clips(all_plays_df, output_folder, extra_lenght= 120)




