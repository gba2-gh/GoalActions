import os
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd


class ClipExtractor:
    def __init__(self):

        self.extraction_resolution= '224p'
        self.extraction_format = 'mkv'
        self.video_root = "D:/soccernetDataset"
        self.clip_lenght = 5 #seconds
        self.by_end = False
        self.only_goals = False
        self.only_no_goals = False



    def extract_video_segment(self, video_path, start_time, end_time, output_path):
        if os.path.exists(output_path):
            print(f'Overwriting: {output_path}')
            #print(f"Skipped: '{output_path}' already exists.")
            #return

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


    def extract_clips(self, df, output_folder):
        for index,play in df.iterrows():
            if self.only_goals and play['label'] == 'False':
                continue
            
            if self.only_no_goals and play['label'] == 'True':
                continue

            if play['half_time'] == 3 or play['half_time'] == 4:
                continue
            #if index > 10:
            # break
            input_path = f"{self.video_root}/{play['league']}/{play['season']}/{play['match']}/{play['half_time']}_{self.extraction_resolution}.{self.extraction_format}"
            output_path = f"{output_folder}\\action_{play['id']}\\clip.mp4" #folder named action to keep consistency with XVARS
            print(f'processing output: {output_path}')
            if self.by_end:
                self.extract_video_segment( input_path, play['time'] - self.clip_lenght, play['time'], output_path)
            else:
                self.extract_video_segment( input_path, play['time'], play['time'] + self.clip_lenght, output_path)
        return

    def extract_extra_clips(self, df, output_folder, extra_lenght = 60):
        for index,play in df.iterrows():
            if self.only_goals and play['label'] == 'False':
                continue
            if self.only_no_goals and play['label'] == 'True':
                continue

            if play['half_time'] == 3 or play['half_time'] == 4:
                continue
            #if index > 10:
            # break
            input_path = f"{self.video_root}/{play['league']}/{play['season']}/{play['match']}/{play['half_time']}_{self.extraction_resolution}.{self.extraction_format}"
            #extra before
            output_path = f"{output_folder}\\action_{play['id']}\\clip_before.mp4" #folder named action to keep consistency with XVARS
            print(f'processing output: {output_path}')
            self.extract_video_segment( input_path, play['time'] - extra_lenght, play['time'], output_path)
            #extra after
            output_path = f"{output_folder}\\action_{play['id']}\\clip_after.mp4" #folder named action to keep consistency with XVARS
            print(f'processing output: {output_path}')
            self.extract_video_segment( input_path, play['time'] + self.clip_lenght , play['time'] + self.clip_lenght + extra_lenght, output_path)

        return

clip_extractor = ClipExtractor()
clip_extractor.only_no_goals = True
output_folder = "results\\video_goal_output_60_no_goals"
all_plays_df = pd.read_csv('results/all_plays.csv')
clip_extractor.extract_clips(all_plays_df, output_folder)
clip_extractor.extract_extra_clips(all_plays_df, output_folder, extra_lenght= 60)




# output_folder = "results\\video_goal_output_60_updated"
# all_plays_df = pd.read_csv('results/all_plays_updated.csv')


# for index,play in all_plays_df.iterrows():
#     if play['half_time'] == 3 or play['half_time'] == 4:
#         continue
#     if index > 60:
#         break
#     if play['updated'] == 'FALSE':
#         continue
#     input_path = f"{video_root}/{play['league']}/{play['season']}/{play['match']}/{play['half_time']}_{extraction_resolution}.{extraction_format}"
    
#     if play['time_label'] == 'NF':
#         print(f'NF. {input_path}')
    
#     output_path = f"{output_folder}\\action_{play['id']}\\clip.mp4" #folder named action to keep consistency with XVARS
#     print(f'processing output: {output_path}')
#     extract_video_segment( input_path, play['end_time'] - clip_lenght, play['end_time'], output_path)
