import pickle  # For serializing and deserializing Python objects, used to save/load data from a file.
import cv2  # OpenCV library for computer vision tasks.
import numpy as np  # NumPy library for numerical operations, such as array manipulations.
import os  # To handle file and directory operations.
import sys  # For interacting with the Python runtime environment.

# Adding the parent directory to the system path to allow importing modules from it.
sys.path.append('../')

# Importing utility functions for measuring distances.
from utils import measure_distance, measure_xy_distance

# Class definition for estimating camera movement between frames.
class CameraMovementEstimator():
    def __init__(self, frame):
        # Define a minimum distance threshold to consider significant camera movement.
        self.minimum_distance = 5

        # Parameters for Lucas-Kanade Optical Flow, which tracks motion between frames.
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the search window at each pyramid level.
            maxLevel=2,  # Maximum number of pyramid levels for the optical flow computation.
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria for the iterative search.
        )

        # Convert the first video frame to grayscale for easier feature tracking.
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask to define the areas where features can be detected.
        mask_features = np.zeros_like(first_frame_grayscale)  # Initialize a mask with zeros (black).
        mask_features[:, 0:20] = 1  # Allow features on the left edge.
        mask_features[:, 900:1050] = 1  # Allow features on the right edge.

        # Parameters for detecting good features to track in the frame.
        self.features = dict(
            maxCorners=100,  # Maximum number of corners to return.
            qualityLevel=0.3,  # Minimum quality of corners.
            minDistance=3,  # Minimum Euclidean distance between corners.
            blockSize=7,  # Size of the neighborhood for corner detection.
            mask=mask_features  # Mask to limit the feature detection area.
        )

    # Adds adjusted positions to tracks based on camera movement.
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # Iterate through tracked objects and their frames.
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']  # Original position of the object.
                    camera_movement = camera_movement_per_frame[frame_num]  # Camera movement for this frame.
                    # Adjust position by subtracting the camera movement.
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    # Estimates camera movement between consecutive frames.
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # If a precomputed stub exists, load camera movement data from it.
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize camera movement as zero for all frames.
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect initial features.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Loop through subsequent frames to calculate camera movement.
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Compute optical flow to find feature movements.
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0  # Initialize max distance for feature movement.
            camera_movement_x, camera_movement_y = 0, 0

            # Iterate through old and new features to calculate movement.
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Measure the distance between old and new feature points.
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:  # Update max distance if a larger one is found.
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # If the max distance exceeds the threshold, update camera movement.
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Detect new features for the next frame.
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            old_gray = frame_gray.copy()  # Update the old frame for the next iteration.

        # Optionally save the computed camera movement to a stub.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    # Draws camera movement information on video frames.
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []  # List to store frames with annotations.

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Copy the frame to avoid modifying the original.

            # Create a semi-transparent overlay for the text box.
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)  # White rectangle.
            alpha = 0.6  # Transparency factor.
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend overlay with the frame.

            # Get camera movement values for this frame.
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Add text annotations for the camera movement.
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)  # Add the annotated frame to the list.

        return output_frames  # Return the frames with annotations.
