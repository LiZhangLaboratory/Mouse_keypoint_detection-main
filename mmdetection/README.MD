# README

black coat mouse (e.g., C57BL/6J) .pth file could be found using the following link:
https://drive.google.com/file/d/18BL_x1vwROe354zW-VZB6l1NAZLeNF20/view?usp=sharing
white coat mouse (e.g., 5xFAD) .pth file could be found using the following link:
https://drive.google.com/file/d/18BL_x1vwROe354zW-VZB6l1NAZLeNF20/view?usp=sharing



## Overview

This Python script processes video files using a pre-trained Faster R-CNN model for object detection. It performs the following steps:

1. Loads video files from the specified folder.
2. Uses the Faster R-CNN model to detect objects in each video frame.
3. Extracts the center coordinates of the bounding boxes (representing detected objects).
4. Draws a red dot at the center of each bounding box on the video frame.
5. Saves the processed video with the red dot annotations and outputs a CSV file containing the frame number and the center coordinates.

## Prerequisites

- **Python 3.x**
- **OpenCV**: For video processing and annotation.
- **Numpy**: For numerical operations.
- **PyTorch and MMDetection**: For running Faster R-CNN inference.
- **CUDA**: If running the model on a GPU.

### Required Python packages:
Install the required dependencies by running:
```bash
pip install opencv-python numpy mmdet
```

You also need to have MMDetection properly installed. For installation instructions, refer to the [MMDetection GitHub repository](https://github.com/open-mmlab/mmdetection).

## Setup

1. **Pre-trained Model & Config:**
   The script is set up to use a Faster R-CNN model with a ResNet-50 backbone. Ensure you have the correct paths to:
   - The configuration file (`.py` file).
   - The pre-trained model weights (`.pth` file).
   
   Modify the following lines in the script with your own paths:
   ```python
   inferencer = DetInferencer(
       model='path/to/config.py',
       weights='path/to/weights.pth',
       device='cuda'
   )
   ```

2. **Folder Structure:**
   Place the video files you want to process in a folder. The script processes `.mp4` files.

## Usage

Run the script from the command line:

```bash
python script_name.py --input_folder <path_to_your_video_folder>
```

### Arguments:
- `--input_folder`: The path to the folder containing the `.mp4` video files you want to process.

### Output:
- For each video processed, the script will generate:
  - A video file with red dot annotations (`_predicted.mp4`).
  - A CSV file containing the frame number, and the X and Y coordinates of the center of the detected objects (`_predicted.csv`).

## Example

```bash
python process_videos.py --input_folder /path/to/videos
```

For example, if there is a video `example.mp4` in your folder:
- The processed video will be saved as `example_predicted.mp4`.
- The CSV file with the center coordinates will be saved as `example_predicted.csv`.

## Additional Information

- **Bounding Box Handling**: For each bounding box detected, the center of the box is calculated and marked with a red dot on the video frame.
- **Device Usage**: The model is set to run on a GPU (`device='cuda'`). If you wish to use the CPU, change the `device` argument to `'cpu'`.

## License

This project is licensed under the MIT License.



