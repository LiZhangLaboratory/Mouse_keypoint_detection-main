import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint  # Import QPoint here
import re
import csv

# Set matplotlib backend
import matplotlib

matplotlib.use('TkAgg')

# use bfloat16 for the entire notebook

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


class InteractiveAnnotation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Annotation")

        # GUI Components
        self.initUI()

        # Initialize video variables
        self.video_dir = ""
        self.frame_idx = 0
        self.frame_names = []

        # Initialize prompts dictionary and ann_frame_idx
        self.prompts = {}
        self.ann_frame_idx = 0  # Selecting points in the first frame
        self.inference_state = None

    def initUI(self):
        # Create layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Button to select input folder
        self.select_folder_btn = QPushButton("Select Input Folder", self)
        self.select_folder_btn.clicked.connect(self.select_input_folder)
        self.layout.addWidget(self.select_folder_btn)

        # Image display label
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        # Annotation variables
        self.points = {1: [], 2: []}  # Dictionary to store points for each object
        self.labels = {1: [], 2: []}  # Dictionary to store labels for each object
        self.ann_obj_id = 1  # Start with object ID 1

        # Show current object ID
        self.statusBar = self.statusBar()
        self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")

    def select_input_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder = QFileDialog.getExistingDirectory(self, "Select Video Frame Directory", options=options)

        if folder:
            self.video_dir = folder
            self.load_video_frames()
            self.frame_idx = 0

            # Initialize the predictor state with the selected video path
            self.inference_state = predictor.init_state(video_path=self.video_dir)

            # Load the first frame and update the display
            self.load_frame(self.frame_idx)
            self.update_image()

    def load_video_frames(self):
        # Get list of frame file names
        self.frame_names = [
            p
            for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        # Extract numbers from file names for sorting
        self.frame_names.sort(key=lambda p: self.extract_number(p))

    def extract_number(self, filename):
        """Extract numbers from a filename for sorting."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')  # Use inf for non-numeric names to sort them last

    def load_frame(self, idx):
        # Load the specific frame by index
        frame_path = os.path.join(self.video_dir, self.frame_names[idx])
        self.frame = Image.open(frame_path)
        self.frame_data = np.array(self.frame)
        self.setFixedSize(self.frame.width, self.frame.height)  # Adjust window size to image size

    def update_image(self):
        # Create a QImage from the frame data
        qimage = QImage(
            self.frame_data.data,
            self.frame_data.shape[1],
            self.frame_data.shape[0],
            self.frame_data.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if self.video_dir == "":
            print("No video directory selected.")
            return

        pos = event.pos()

        # Use the actual image size for coordinates
        x = int(pos.x())
        y = int(pos.y())

        if event.button() == Qt.LeftButton:
            self.points[self.ann_obj_id].append([x, y])
            self.labels[self.ann_obj_id].append(1)
            print(f"Added positive point for Object ID {self.ann_obj_id}: ({x}, {y})")
        elif event.button() == Qt.RightButton:
            self.points[self.ann_obj_id].append([x, y])
            self.labels[self.ann_obj_id].append(0)
            print(f"Added negative point for Object ID {self.ann_obj_id}: ({x}, {y})")

        # Draw the point directly on the QPixmap
        self.draw_point_on_image(x, y, positive=(event.button() == Qt.LeftButton))

    def draw_point_on_image(self, x, y, positive=True):
        # Create a QPixmap from the existing image
        pixmap = self.image_label.pixmap()

        # Create a QPainter object to draw on the QPixmap
        painter = QPainter(pixmap)
        color = QColor("green") if positive else QColor("red")
        painter.setPen(color)
        painter.setBrush(color)

        # Draw a circle at the specified position
        point_radius = 5
        painter.drawEllipse(QPoint(x, y), point_radius, point_radius)

        # End painting
        painter.end()

        # Update the QLabel with the new QPixmap
        self.image_label.setPixmap(pixmap)

    def redraw_points(self):
        # No need to redraw all points; they are added directly now
        pass

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_1:
            self.ann_obj_id = 1
            self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")
            print("Switched to Object ID 1")
        elif key == Qt.Key_2:
            self.ann_obj_id = 2
            self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")
            print("Switched to Object ID 2")
        elif key == Qt.Key_S:
            self.save_points()
            self.propagate_annotations()  # Start propagation after saving points
        elif key == Qt.Key_Q:
            self.close()

    def save_points(self):
        for obj_id in [1, 2]:
            points = np.array(self.points[obj_id], dtype=np.float32)
            labels = np.array(self.labels[obj_id], dtype=np.int32)

            if len(points) > 0:
                self.prompts[obj_id] = points, labels
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

                # Display the result for the current object
                plt.figure(figsize=(12, 8))
                plt.title(f"Object ID {obj_id} - frame {self.ann_frame_idx}")
                plt.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.ann_frame_idx])))
                show_points(points, labels, plt.gca())
                for i, out_obj_id in enumerate(out_obj_ids):
                    show_points(*self.prompts[out_obj_id], plt.gca())
                    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
                plt.show()  # Show results for the current object

                print(f"Points for Object ID {obj_id} saved.")

    def propagate_annotations(self):
        output_dir = os.path.join(self.video_dir, "annotated_frames")
        os.makedirs(output_dir, exist_ok=True)

        video_path = os.path.join(output_dir, "annotated_video.mp4")

        frame_height, frame_width = self.frame_data.shape[:2]
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
                self.inference_state
            ):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

                # Annotate and save each frame
                annotated_frame = self.annotate_frame(out_frame_idx, video_segments[out_frame_idx])
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

    def annotate_frame(self, frame_idx, segments):
        # Annotate a frame with the segmentation masks and save it
        frame_path = os.path.join(self.video_dir, self.frame_names[frame_idx])
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


# Run the application
app = QApplication(sys.argv)
window = InteractiveAnnotation()
window.show()
sys.exit(app.exec_())
