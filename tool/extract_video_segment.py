
import os
import cv2
from moviepy.editor import VideoFileClip
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path_video',   required=True, type=str, help='Path to video' )
    parser.add_argument('--clip_start_time',   required=True, type=float, help='New clip start time' )
    parser.add_argument('--clip_end_time',   required=True, type=float, help='New clip end time' )


    args = parser.parse_args()
    path_video = args.path_video
    parts = path_video.split("_")
    match = "_".join(parts[3:])

    clip_start_time = args.clip_start_time *60
    clip_end_time = args.clip_end_time *60

    output_path = f'results\\output_video_segment\\{match}\\clip_{clip_start_time}_{clip_end_time}.mp4'

    extract_video_segment(path_video, clip_start_time, clip_end_time, output_path)
    #extract_video_segment_ffmpeg(path_video, clip_start_time, clip_end_time, output_path)