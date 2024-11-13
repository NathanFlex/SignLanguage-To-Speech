import os
from pathlib import Path
import cv2
def count_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def process_directory(directory_path):
    directory_path = Path(directory_path)
    for video_file in directory_path.glob('*.mp4'):
        frame_count = count_frames(video_file)
        print(f"{video_file.name}: {frame_count} frames")

process_directory(r"C:\Users\HP\PycharmProjects\VIT\Project\videos")