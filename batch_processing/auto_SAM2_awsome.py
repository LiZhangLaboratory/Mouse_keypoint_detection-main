import os
import numpy as np
import cv2
import pandas as pd
from annotation_module import process_video_folder  # Import the function from the previous script

def read_last_annotations(csv_path, image_shape):
    df = pd.read_csv(csv_path)
    print("Column names in the CSV file:", df.columns)  # Print column names for debugging

    # Read the last two rows for object IDs and mask information
    last_two_rows = df.tail(2)
    points = {1: None, 2: None}
    contours_dict = {1: None, 2: None}

    # Correct column names
    object_id_column = 'Object ID'
    mask_info_column = 'Mask Data'

    if object_id_column not in df.columns or mask_info_column not in df.columns:
        raise ValueError(f"CSV file does not contain required columns: '{object_id_column}', '{mask_info_column}'")

    height, width = image_shape

    # Extract the central points from the mask information
    for idx, row in last_two_rows.iterrows():
        object_id = row[object_id_column]
        mask_info = row[mask_info_column]

        # Convert mask_info to numpy array if possible
        try:
            mask_info_array = np.array(eval(mask_info)).astype(bool)
            if mask_info_array.ndim == 1 and mask_info_array.size == height * width:
                # Reshape the mask to the original image dimensions
                mask_info_reshaped = mask_info_array.reshape((height, width))

                # Convert boolean mask to uint8
                mask_info_uint8 = mask_info_reshaped.astype(np.uint8) * 255

                # Find contours
                contours, _ = cv2.findContours(mask_info_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    contours_dict[object_id] = contours
                    
                    # Calculate the centroid of the largest contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        central_point = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                        points[object_id] = central_point
                        print(f"Central point for object {object_id}: {central_point}")  # Debug statement
                    else:
                        print(f"Contour area is zero for object {object_id}")
                else:
                    print(f"No contours found for object {object_id}")
            else:
                print(f"Invalid mask info format for object {object_id}: {mask_info_array.shape}")
        except Exception as e:
            print(f"Error processing mask info for object {object_id}: {e}")

    return points, contours_dict

def apply_annotations_to_next_folder(next_folder, central_points, contours_dict):
    first_image_path = os.path.join(next_folder, sorted(os.listdir(next_folder))[0])
    image = cv2.imread(first_image_path)

    points = {1: [], 2: []}
    labels = {1: [], 2: []}

    # Use central points to create positive and negative annotations
    points[1].append(central_points[1])
    labels[1].append(1)  # Positive for Object 1
    points[1].append(central_points[2])
    labels[1].append(0)  # Negative for Object 1

    points[2].append(central_points[2])
    labels[2].append(1)  # Positive for Object 2
    points[2].append(central_points[1])
    labels[2].append(0)  # Negative for Object 2

    # Convert to numpy arrays
    annotations = {
        1: (np.array(points[1], dtype=np.float32), np.array(labels[1], dtype=np.int32)),
        2: (np.array(points[2], dtype=np.float32), np.array(labels[2], dtype=np.int32)),
    }

    # Process the video folder with the annotations
    process_video_folder(next_folder, annotations, csv_file=None)

    # Draw the mask contours and central points
    #for object_id, point in central_points.items():
     #   if point is not None:
      #      if object_id == 1:
       #         color = (0, 255, 0)  # Green for object 1
        #    else:
         #       color = (0, 0, 255)  # Red for object 2
#
 #           cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)
  #          cv2.putText(image, f"Object {object_id}", (int(point[0]) + 10, int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw contours
   #         if contours_dict[object_id] is not None:
    #            cv2.drawContours(image, contours_dict[object_id], -1, color, 2)

    # Display the image
    #cv2.imshow('Annotated Image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Define the root directory
root_dir = 'D:/batched_images/2022-07-29 19-10-18_cropped'
initial_folder = 'batch18'
next_folders = [f'batch{i}' for i in range(19, 51)]

# Assuming you know the dimensions of the images
image_shape = (350, 500)  # Replace with actual height and width of the images

# Read the last annotations from the initial folder
csv_path = os.path.join(root_dir, initial_folder, 'annotated_frames', 'mask_information.csv')
central_points, contours_dict = read_last_annotations(csv_path, image_shape)

# Process each subsequent folder
for folder in next_folders:
    next_folder_path = os.path.join(root_dir, folder)
    if os.path.exists(next_folder_path):
        apply_annotations_to_next_folder(next_folder_path, central_points, contours_dict)
        # Update the central points for the next iteration
        csv_path = os.path.join(next_folder_path, 'annotated_frames', 'mask_information.csv')
        central_points, contours_dict = read_last_annotations(csv_path, image_shape)
    else:
        print(f"Folder {next_folder_path} does not exist.")
