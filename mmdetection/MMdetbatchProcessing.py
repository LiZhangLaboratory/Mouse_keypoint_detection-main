import os
import cv2
import numpy as np
import csv
import argparse
from mmdet.apis import DetInferencer

# Initialize the Inferencer with your Faster R-CNN config and weights
inferencer = DetInferencer(model=r'I:\Apathy_5xFAD\mmdetection/faster-rcnn_r50-caffe_fpn_ms-1x_coco-mouse.py',
                           weights=r'I:\Apathy_5xFAD\mmdetection\faster-rcnn_r50-caffe_fpn_ms-1x_coco-whiteMouse/epoch_200.pth', 
                           device='cuda')

# Function to process a single video file
def process_video(video_path, output_video_path, csv_output_path):
    # Open video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer to save the output video with dots
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Prepare to save CSV data
    with open(csv_output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame', 'Center X', 'Center Y'])

        frame_number = 0
        # Loop through the video frame by frame
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            if not ret:
                break  # End of video
            
            # Inference on the current frame using Faster R-CNN
            results = inferencer(frame)

            # Extract bounding boxes from the results
            for bbox in results['predictions'][0]['bboxes']:
                min_x, min_y, max_x, max_y = bbox
                center_x = int((min_x + max_x) / 2)
                center_y = int((min_y + max_y) / 2)
                
                # Write the center coordinates to the CSV file
                csv_writer.writerow([frame_number, center_x, center_y])

                # Draw the red dot at the center of the bbox on the frame
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Write the frame with the red dot into the output video
            video_writer.write(frame)
            
            frame_number += 1

    # Release video capture and writer
    video_capture.release()
    video_writer.release()

    print(f'Processed video: {video_path}')
    print(f'CSV saved to: {csv_output_path}')
    print(f'Video saved to: {output_video_path}')

# Main function to process all videos in a folder
def main(input_folder):
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(input_folder, filename)

            # Define output paths based on the input file name
            output_video_path = os.path.join(input_folder, filename.replace('.mp4', '_predicted.mp4'))
            csv_output_path = os.path.join(input_folder, filename.replace('.mp4', '_predicted.csv'))

            # Process the video
            process_video(video_path, output_video_path, csv_output_path)

if __name__ == '__main__':
    # Argument parsing to get the folder input from the command line
    parser = argparse.ArgumentParser(description='Process videos in a folder and save predictions.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the folder containing video files')

    args = parser.parse_args()

    # Call the main function with the input folder
    main(args.input_folder)
