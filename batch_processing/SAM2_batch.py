import sys
import os
import numpy as np
import re
import cv2
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QMessageBox,
    QListWidget,
    QLabel
)
from PyQt5.QtCore import Qt
from annotation_module import process_video_folder  # Import the function from the previous script

class InteractivePointSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Point Selector")

        # Initialize GUI components
        self.initUI()

        # Initialize video variables
        self.video_dirs = []  # Store multiple directories
        self.current_folder_idx = 0
        self.frame_idx = 0
        self.frame_names = []

        # Annotation variables
        self.points = {1: [], 2: []}  # Dictionary to store points for each object
        self.labels = {1: [], 2: []}  # Dictionary to store labels for each object
        self.annotations = {}  # Store annotations for each folder
        self.ann_obj_id = 1  # Start with object ID 1

        # Show current object ID
        self.statusBar = self.statusBar()
        self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")

        # OpenCV Window Name
        self.window_name = "Image Annotation"

    def initUI(self):
        # Create layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Button to select input folders
        self.select_folder_btn = QPushButton("Select Input Folders", self)
        self.select_folder_btn.setFixedHeight(40)  # Fix button height
        self.select_folder_btn.clicked.connect(self.select_input_folders)
        self.layout.addWidget(self.select_folder_btn)

        # List widget to display selected folders
        self.folder_list = QListWidget(self)
        self.folder_list.setFixedHeight(100)  # Fix folder list height
        self.layout.addWidget(self.folder_list)

        # Button to proceed to next folder
        self.next_folder_btn = QPushButton("Next Folder", self)
        self.next_folder_btn.setFixedHeight(40)  # Fix button height
        self.next_folder_btn.clicked.connect(self.next_folder)
        self.layout.addWidget(self.next_folder_btn)

        # Button to process all folders
        self.process_btn = QPushButton("Process All Folders", self)
        self.process_btn.setFixedHeight(40)  # Fix button height
        self.process_btn.clicked.connect(self.process_folders)
        self.layout.addWidget(self.process_btn)

        # Status label
        self.status_label = QLabel("Status: Ready", self)
        self.layout.addWidget(self.status_label)

    def select_input_folders(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        self.video_dirs = []  # Reset the list of directories
        self.folder_list.clear()  # Clear the list widget

        while True:
            folder = QFileDialog.getExistingDirectory(self, "Select Video Frame Directory", options=options)
            if folder:
                self.video_dirs.append(folder)
                self.folder_list.addItem(folder)  # Add folder to the list widget
            else:
                break

        if not self.video_dirs:
            QMessageBox.warning(self, "No Folders Selected", "Please select at least one folder.")
        else:
            self.current_folder_idx = 0
            self.load_video_frames(self.video_dirs[self.current_folder_idx])
            self.frame_idx = 0
            self.load_frame(self.frame_idx)
            self.display_image()

    def next_folder(self):
        """Save the current folder's points and move to the next folder."""
        if self.current_folder_idx < len(self.video_dirs):
            # Save the annotations for the current folder
            self.save_points()

            # Move to the next folder
            self.current_folder_idx += 1

            if self.current_folder_idx < len(self.video_dirs):
                self.load_video_frames(self.video_dirs[self.current_folder_idx])
                self.frame_idx = 0
                self.load_frame(self.frame_idx)
                self.display_image()
                self.points = {1: [], 2: []}  # Reset points for new folder
                self.labels = {1: [], 2: []}
            else:
                print("All folders have been annotated.")
        else:
            print("There are no more folders to annotate.")

    def load_video_frames(self, video_dir):
        # Get list of frame file names
        self.frame_names = [
            p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        # Extract numbers from file names for sorting
        self.frame_names.sort(key=lambda p: self.extract_number(p))

    def extract_number(self, filename):
        """Extract numbers from a filename for sorting."""
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')  # Use inf for non-numeric names to sort them last

    def load_frame(self, idx):
        # Load the specific frame by index
        frame_path = os.path.join(self.video_dirs[self.current_folder_idx], self.frame_names[idx])
        self.frame = cv2.imread(frame_path)

    def display_image(self):
        # Ensure the window uses the raw image dimensions
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.frame)

        # Set mouse callback for annotation
        cv2.setMouseCallback(self.window_name, self.annotate_image)

        # Listen for key presses to switch object IDs
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                self.ann_obj_id = 1
                self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")
                print("Switched to Object ID 1")
            elif key == ord('2'):
                self.ann_obj_id = 2
                self.statusBar.showMessage(f"Annotating Object ID: {self.ann_obj_id}")
                print("Switched to Object ID 2")
            elif key == ord('s'):
                self.save_points()
                print("Points saved.")
                break
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                break

    def annotate_image(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add positive point
            self.points[self.ann_obj_id].append([x, y])
            self.labels[self.ann_obj_id].append(1)
            print(f"Added positive point for Object ID {self.ann_obj_id}: ({x}, {y})")
            self.draw_point_on_image(x, y, positive=True)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add negative point
            self.points[self.ann_obj_id].append([x, y])
            self.labels[self.ann_obj_id].append(0)
            print(f"Added negative point for Object ID {self.ann_obj_id}: ({x}, {y})")
            self.draw_point_on_image(x, y, positive=False)

    def draw_point_on_image(self, x, y, positive=True):
        color = (0, 255, 0) if positive else (0, 0, 255)
        label = "Positive" if positive else "Negative"

        # Draw circle and label on the image
        cv2.circle(self.frame, (x, y), 5, color, -1)
        cv2.putText(self.frame, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Update the OpenCV window
        cv2.imshow(self.window_name, self.frame)

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
        elif key == Qt.Key_Q:
            self.close()

    def save_points(self):
        """Save the selected points and labels for both objects for the current folder."""
        current_folder = self.video_dirs[self.current_folder_idx]
        self.annotations[current_folder] = {
            1: (np.array(self.points[1], dtype=np.float32), np.array(self.labels[1], dtype=np.int32)),
            2: (np.array(self.points[2], dtype=np.float32), np.array(self.labels[2], dtype=np.int32)),
        }
        print(f"Points saved for folder: {current_folder}")

    def process_folders(self):
        """Process all selected folders with the stored points."""
        # Ensure all folders are annotated
        if not self.annotations:
            QMessageBox.warning(self, "No Annotations", "Please annotate points for each folder before processing.")
            return

        # Process each selected folder
        for folder in self.video_dirs:
            if folder in self.annotations:
                print(f"Processing folder: {folder}")
                process_video_folder(folder, self.annotations[folder], csv_file=None)
            else:
                print(f"No annotations found for folder: {folder}")

# Run the application
app = QApplication(sys.argv)
window = InteractivePointSelector()
window.show()
sys.exit(app.exec_())
