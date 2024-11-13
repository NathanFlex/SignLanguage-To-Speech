import cv2
import numpy as np
from pathlib import Path
import json
import pandas

csv_file = r'Video_Records.csv'
df = pandas.read_csv(csv_file)
labels = []

def load_video(video_path, target_frames=30, target_size=(112, 112)):
    cap = cv2.VideoCapture(str(video_path))
    filename = str(video_path)[-1:-10:-1]
    filename = filename[-1:-(len(filename)-3):-1]
    label = df[df['id'] == int(filename)]
    label = label.values[0][1]
    labels.append(label)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    if total_frames < target_frames:
        raise ValueError("Not enough frames")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    frames = np.array(frames)
    if len(frames) == target_frames:
        return frames

    elif len(frames) > target_frames:
        #sample uniformly
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        return frames[indices]

    else:
        indices = np.mod(np.arange(target_frames), len(frames))
        return frames[indices]


def prepare_video_batch(video_paths, target_frames=30, target_size=(112, 112)):
    videos = []
    for video_path in video_paths:
        try:
            video = load_video(video_path, target_frames, target_size)
            video = np.transpose(video, (3,0,1,2))
            videos.append(video)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue
    return np.stack(videos, axis=0)

def process_video_dataset(video_dir):
    video_extensions = ['.mp4']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(Path(video_dir).glob(f'*{ext}'))
    data_generator = prepare_video_batch(
        video_paths,
        target_frames=30,
        target_size=(112, 112),
    )

    return data_generator


def save_numpy(video_data, save_dir, labels, filename="video_data"):
    data_path = save_dir + "/" + f"{filename}.npy"
    np.save(data_path, video_data)

    metadata = {
        'shape': video_data.shape,
        'dtype': str(video_data.dtype),
        'has_labels': labels is not None
    }
    metadata_path = save_dir + "/" + f"{filename}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

video_dir = r'C:\Users\HP\PycharmProjects\VIT\Project\videos'
data_generator = process_video_dataset(video_dir)
#save_numpy(data_generator,r'C:\Users\HP\PycharmProjects\VIT\Project\VideoData',labels)
