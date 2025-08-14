import os
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd

extraction_resolution= '224p'
exrtaction_format = 'mkv'

def extract_video_segment(video_path, start_time, end_time, output_path):
    if os.path.exists(output_path):
        print(f"Skipped: '{output_path}' already exists.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    clip = VideoFileClip(video_path)
    duration = clip.duration

    if start_time >= duration:
        print(f"Skipped: start_time ({start_time}s) >= video duration ({duration:.2f}s)")
        return

    # Clamp end_time to the video duration
    end_time = min(end_time, duration)
    clip = VideoFileClip(video_path).subclip(start_time, end_time)
    clip.write_videofile(output_path, codec="libx264", audio=False)  # Output is .mp4 by default


def extract_from_df(df, output_folder):
    for index,play in df.iterrows():
        if index > 10:
            break
        input_path = f"{play['video_path']}/{play['match_ht']}_{extraction_resolution}.{exrtaction_format}"
        output_path = f"{output_folder}\\action_{play['id']}\\clip.mp4" #folder named action to keep consistency with XVARS
        print(f'processing output: {output_path}')
        extract_video_segment( input_path, play['play_start']-15, play['play_end']+15, output_path)
        output_path = f"{output_folder}\\action_{play['id']}\\clip_MT.mp4"
        extract_video_segment(input_path, play['play_start_MT'], play['play_end_MT']+15, output_path)
    return

output_folder = "results\\video_goal_output"
all_plays_df = pd.read_csv('results/all_plays_MT.csv')
extract_from_df(all_plays_df, output_folder)


