import os
import csv
import cv2
from tqdm import tqdm

def parse_annotation_row(row):
    """Parse a single row of the annotation CSV"""
    try:
        video_name = row[0].strip()
        total_frames = int(row[1])  # Not used but parsed for validation
        frame_numbers = [int(num.strip()) for num in row[2:] if num.strip().isdigit()]
        return video_name, frame_numbers
    except (IndexError, ValueError):
        return None, []

def extract_frames(root_dir, set_names):
    for set_name in set_names:
        csv_path = os.path.join(root_dir, 'annotations', set_name, f"{set_name}_annotated_frames.csv")
        
        # Read raw CSV lines
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in tqdm(reader, desc=f"Processing {set_name}"):
                if len(row) < 3:  # Minimum: video_name, total_frames, at least 1 frame
                    continue
                
                video_name, frame_nums = parse_annotation_row(row)
                if not video_name or not frame_nums:
                    continue
                
                video_path = os.path.join(root_dir, 'PIE_clips', set_name, f"{video_name}.mp4")
                output_dir = os.path.join(root_dir, 'extracted_frames', set_name, video_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Extract frames
                cap = cv2.VideoCapture(video_path)
                for fn in frame_nums:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
                        cv2.imwrite(
                            os.path.join(output_dir, f"frame_{fn:05d}.jpg"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85]
                        )
                cap.release()

if __name__ == "__main__":
    extract_frames(
        root_dir="/mnt/d/Bachelor/PIE",
        set_names=["set03"]
    )