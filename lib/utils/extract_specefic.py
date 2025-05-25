import cv2
import os

def extract_specific_frames(video_path, frame_indices, output_dir):
    """
    Extracts and saves specific frames from a video.

    Args:
        video_path (str): Path to the video file.
        frame_indices (list): List of frame indices (integers) to extract.
        output_dir (str): Directory to save extracted frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for idx in frame_indices:
        if idx >= total_frames:
            print(f"Warning: Frame index {idx} exceeds total frames ({total_frames}). Skipping.")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()

        if success:
            output_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {idx} to {output_path}")
        else:
            print(f"Failed to read frame {idx}")

    cap.release()
    print("Done extracting frames.")

# Example usage:
video_path = "/mnt/d/Bachelor/video_0001_traj_pred1.mp4"  # replace with your video path
frame_indices = [3752,3782,3802,3832,3862]  # frames you want to extract
output_dir = "/mnt/d/Bachelor/extracted_frames_VIS"  # replace with your desired output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(video_path):
    print(f"Video path {video_path} does not exist.")

extract_specific_frames(video_path, frame_indices, output_dir)
