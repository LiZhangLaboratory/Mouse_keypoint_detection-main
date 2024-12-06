import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import re
import csv

# Set matplotlib backend
import matplotlib

matplotlib.use('TkAgg')

# Use bfloat16 for the entire script
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def process_video_folder(video_dir, points_dict, csv_file=None):
    """
    Process a folder containing video frames and perform annotation using given points.

    Parameters:
        video_dir (str): Path to the directory containing video frames.
        points_dict (dict): Dictionary containing points and labels for two objects.
                            Format: {1: (points, labels), 2: (points, labels)}
        csv_file (str): Path to a CSV file containing points and labels (optional).
    """

    # Initialize video variables
    frame_idx = 0
    frame_names = []

    # Initialize prompts dictionary and ann_frame_idx
    prompts = {}
    ann_frame_idx = 0  # Selecting points in the first frame
    inference_state = predictor.init_state(video_path=video_dir)

    # Load CSV data if provided
    if csv_file:
        csv_points, csv_labels = load_csv_data(csv_file)
        if csv_points is not None and csv_labels is not None:
            # Use CSV data directly if available
            points_dict = {1: (csv_points[1], csv_labels[1]), 2: (csv_points[2], csv_labels[2])}

    # Load video frames
    frame_names = [
        p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: extract_number(p))

    # Load the first frame
    frame_data = load_frame(video_dir, frame_names, frame_idx)

    # Annotate and save points
    for obj_id in [1, 2]:
        points, labels = points_dict[obj_id]
        if len(points) > 0:
            prompts[obj_id] = points, labels
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

            # Display the result for the current object
            plt.figure(figsize=(12, 8))
            plt.title(f"Object ID {obj_id} - frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            for i, out_obj_id in enumerate(out_obj_ids):
                show_points(*prompts[out_obj_id], plt.gca())
                show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
            plt.show()

            print(f"Points for Object ID {obj_id} saved.")

    # Propagate annotations
    propagate_annotations(video_dir, frame_data, inference_state, frame_names)


def load_csv_data(csv_file):
    """Load points and labels from CSV file."""
    csv_points = {1: [], 2: []}
    csv_labels = {1: [], 2: []}
    try:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                obj_id = int(row[0])
                frame_idx = int(row[1])
                points = eval(row[2])  # Convert string representation of list back to list
                labels = eval(row[3])
                if frame_idx == len(frame_names) - 1:  # Last frame
                    csv_points[obj_id].extend(points)
                    csv_labels[obj_id].extend(labels)
        print("CSV data loaded successfully.")
    except Exception as e:
        print(f"Error loading CSV file: {e}")

    return csv_points, csv_labels


def extract_number(filename):
    """Extract numbers from a filename for sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # Use inf for non-numeric names to sort them last


def load_frame(video_dir, frame_names, idx):
    """Load the specific frame by index."""
    frame_path = os.path.join(video_dir, frame_names[idx])
    frame = Image.open(frame_path)
    frame_data = np.array(frame)
    return frame_data


def propagate_annotations(video_dir, frame_data, inference_state, frame_names):
    output_dir = os.path.join(video_dir, "annotated_frames")
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "annotated_video.mp4")

    frame_height, frame_width = frame_data.shape[:2]
    video_writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height)
    )

    # CSV file setup
    csv_path = os.path.join(output_dir, "mask_information.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Object ID", "Frame", "Mask Data"])  # CSV headers

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # Annotate and save each frame
            annotated_frame = annotate_frame(video_dir, frame_names, out_frame_idx, video_segments[out_frame_idx])
            frame_path = os.path.join(output_dir, f"frame_{out_frame_idx:03}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            # Write to video
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            else:
                print("Video writer is not initialized properly.")

            # Export mask data to CSV
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                flattened_mask = out_mask.flatten()
                writer.writerow([out_obj_id, out_frame_idx, flattened_mask.tolist()])

    video_writer.release()
    print(f"Propagation completed. Video saved at {video_path}")
    print(f"Mask data saved to {csv_path}")


def annotate_frame(video_dir, frame_names, frame_idx, segments):
    """Annotate a frame with the segmentation masks and save it."""
    frame_path = os.path.join(video_dir, frame_names[frame_idx])
    frame = cv2.imread(frame_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for out_obj_id, out_mask in segments.items():
        show_mask(out_mask, ax, obj_id=out_obj_id)

    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.close(fig)

    # Convert Matplotlib figure to image array
    fig.canvas.draw()
    annotated_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_image = annotated_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return annotated_image

