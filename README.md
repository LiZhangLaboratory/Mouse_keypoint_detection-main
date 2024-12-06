Here’s a more substantiated version of the README file, providing additional details and context:

# Mouse Behavioral Annotation

This repository focuses on the development and implementation of a comprehensive pipeline designed to annotate mouse behaviors efficiently. The pipeline leverages a combination of software tools for behavior classification, keypoint detection, bounding box annotation, and segmentation, providing a streamlined process for detecting and tracking behaviors in mice, which are critical for various behavioral neuroscience studies.

## Step 1: BORIS Annotation
BORIS (Behavioral Observation Research Interactive Software) is a versatile and open-source tool used for behavioral annotations. It allows researchers to define and annotate complex behaviors using customizable templates. The first step in the pipeline involves annotating mouse behaviors using BORIS.

- **Download BORIS**: The software can be downloaded from the official BORIS website at [BORIS Software](https://www.boris.unito.it/).
- **User Guide**: A detailed user guide for setting up and using BORIS is available [here](https://www.boris.unito.it/user_guide/pdf/boris_user_guide.pdf#page=3.14). This guide walks users through the interface, configuration, and how to record and analyze behavior annotations.
  
Using BORIS, you can define specific behaviors, including sniffing, grooming, rearing, and more, and synchronize them with video data for precise behavioral labeling.

## Step 2: LabelMe Annotation
LabelMe is a popular annotation tool used primarily for object detection, allowing for bounding box (bbox) and keypoint annotations. In our pipeline, we utilize LabelMe to annotate visual elements in mouse behavior videos, marking specific body parts and poses for subsequent analysis.

- **Download LabelMe**: You can download LabelMe from [GitHub](https://github.com/labelmeai/labelme).
- **Detailed Instructions**: The official documentation on how to use LabelMe for bounding box and keypoint annotations can be found on the GitHub repository page. 

Once the annotations are complete, the next step involves converting these annotations into COCO format for integration with the rest of the pipeline:
- **COCO Conversion**: We use the `labelme2coco` package to convert the annotations into COCO format for compatibility with deep learning models. Find the tool at [labelme2coco](https://github.com/fcakyon/labelme2coco).

## Step 3: Training a Faster R-CNN Mouse Detection Head
We utilize a Faster R-CNN model, a robust and widely used object detection architecture, to detect and classify mouse behaviors. This model is trained using the openmmlab library on a GPU machine for efficient processing of high-resolution videos.

- **Virtual Environment Setup**: First, create a Python virtual environment for model training.
  ```bash
  conda create --name openmmlab python=3.8 -y
  conda activate openmmlab
  ```

- **PyTorch Installation**: Install PyTorch and torchvision using the following command.
  ```bash
  conda install pytorch torchvision -c pytorch
  ```

- **Install OpenMMLab Packages**: 
  ```bash
  pip install -U openmim
  mim install mmengine
  mim install "mmcv>=2.0.0"
  ```

- **Install MMDetection**: Clone the mmdetection repository and set it up.
  ```bash
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  pip install -v -e .
  ```

The training configuration file can be found within this repository. We used labeled data to train the detection head. You can download the trained detection model [here](https://drive.google.com/drive/folders/1xghVPv2haytx1HxOOl0aD77w-fFzncdY?usp=sharing).

## Step 4: Training an HRNet Keypoint Detection Model
HRNet (High-Resolution Network) is used for keypoint detection, which allows us to precisely identify and track keypoints on the mouse body, such as limbs, tail, and head. This step is critical for detailed pose estimation.

## Step 5: SAM + FasterRCNN Based Tracking Model
### SAM2_Tutorial

We incorporate SAM2 (Segment Anything Model v2) along with Faster R-CNN to create a powerful, zero-shot tracking model capable of identifying and segmenting multiple mice simultaneously.

- **Setup Anaconda and CUDA**: Before proceeding, ensure your machine has a GPU and CUDA installed for efficient processing. First, download Anaconda [here](https://www.anaconda.com/download/success) and CUDA 11.8 [here](https://developer.nvidia.com/cuda-11-8-0-download-archive).
  
  Check the CUDA version by typing:
  ```bash
  nvidia-smi
  ```

- **Create Virtual Environment**:
  ```bash
  conda create -n SAM2_test python=3.11
  conda activate SAM2_test
  ```

- **Install SAM2**: Clone the SAM2 repository and install necessary dependencies.
  ```bash
  git clone https://github.com/facebookresearch/segment-anything-2.git
  cd segment-anything-2
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
  pip install hydra-core tqdm matplotlib opencv-python ninja imageio
  ```

- **Build and Test SAM2**: Navigate to the segment-anything-2 directory and build the SAM2 package.
  ```bash
  python setup.py build_ext --inplace
  ```
  
  Test the installation by entering Python and importing SAM2:
  ```bash
  python
  import sam2
  ```

- **Download SAM2 Model**: Download the pre-trained SAM2 large model [here](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) and place it in the `checkpoints` folder.

The SAM2_2obj_test_qt_GUI_csv_v3.py script allows tracking and segmentation of two mice simultaneously with zero-shot capabilities.

## Step 6: Differentiation of Sniffing and Grooming Behavior
In this step, we leverage RAFT (Recurrent All Pairs Field Transforms), to differentiate between mouse behaviors such as sniffing and grooming. Optical flow measures the motion of objects between consecutive frames in a video, providing valuable insight into subtle differences in movement patterns that distinguish specific behaviors.
details see https://github.com/princeton-vl/RAFT
Running RAFT on Video Frames: You can demo the pretrained RAFT model on a sequence of frames to generate optical flow maps:

```bash
python demo.py --model=models/raft-things.pth --path=demo-frames
```
The optical flow data generated using RAFT can differentiate subtle movements such as head bobbing during sniffing or repetitive body motions during grooming, providing a detailed characterization of mouse behavior. The sniffing would not cause a significant body motion while grooming would. 



## Step 7: Verification by Human Annotator
Finally, human annotators review the results produced by the pipeline. This ensures that the automated behavior classification matches human observations, serving as the final layer of validation.

This pipeline offers a modular, scalable solution to mouse behavior annotation, combining the power of deep learning, computer vision, and manual curation for precise behavioral studies.
