from ultralytics import YOLO  # Import YOLO (You Only Look Once) for object detection from ultralytics.
import supervision as sv  # Import Supervision library for tracking objects.
import pickle  # Import pickle for saving and loading data.
import os  # For file path and directory operations.
import numpy as np  # Import NumPy for array and numerical operations.
import pandas as pd  # Import pandas for handling data in DataFrame format.
import cv2  # Import OpenCV for image and video processing.

import sys  # Import sys for system-specific parameters.
sys.path.append('../')  # Add parent directory to path for module imports.
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # Import custom utility functions for handling bounding boxes.

# Define the Tracker class to track and manage objects such as players, referees, and the ball in frames.
class Tracker:
    def __init__(self, model_path):
        # Initialize the YOLO model and the ByteTrack object tracker.
        self.model = YOLO(model_path)  # Load the YOLO model for object detection.
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack for object tracking.

    # Method to add the position of tracked objects to the track records.
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']  # Get the bounding box for the tracked object.
                    # Depending on the object type, get the appropriate position.
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)  # Get center of ball bounding box.
                    else:
                        position = get_foot_position(bbox)  # Get foot position for players and referees.
                    # Add the position to the track information.
                    tracks[object][frame_num][track_id]['position'] = position

    # Method to interpolate missing ball positions for smoother tracking.
    def interpolate_ball_positions(self, ball_positions):
        # Extract bounding box coordinates for each ball position.
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values and fill forward/backward.
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # Reformat the interpolated data into the original format.
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    # Method to detect objects in video frames using the YOLO model.
    def detect_frames(self, frames):
        batch_size = 20  # Define the batch size for detection processing.
        detections = []  # Initialize a list to store the detection results.
        for i in range(0, len(frames), batch_size):
            # Predict object detections on a batch of frames with confidence threshold set to 0.1.
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += detections_batch  # Accumulate detections across batches.
        return detections

    # Method to get the tracks of objects (players, referees, ball) across frames.
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # If requested, load tracks from a saved file (stub) to avoid reprocessing.
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Detect objects in all frames.
        detections = self.detect_frames(frames)

        # Initialize tracks for players, referees, and the ball.
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Process each frame's detection results.
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get class names of detected objects.
            cls_names_inv = {v: k for k, v in cls_names.items()}  # Reverse the class names for quick lookup.

            # Convert detections to a format understood by the supervision library.
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert "goalkeeper" class to "player" for uniform tracking.
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects across frames.
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Initialize empty dictionaries for the current frame's objects.
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Store the tracked object information for players and referees.
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # Get bounding box coordinates.
                cls_id = frame_detection[3]  # Get the class ID.
                track_id = frame_detection[4]  # Get the unique track ID.

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Store the tracked ball information.
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Optionally save the tracks to a stub file.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # Method to draw an ellipse around a tracked object.
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # Get the bottom y-coordinate of the bounding box.
        x_center, _ = get_center_of_bbox(bbox)  # Get the center x-coordinate of the bounding box.
        width = get_bbox_width(bbox)  # Get the width of the bounding box.

        # Draw the ellipse on the frame.
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),  # Adjust height to width ratio.
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Draw a rectangle with track ID if provided.
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            # Draw the rectangle filled with color and annotated with the track ID.
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            # Adjust text position based on track ID size.
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # Add track ID text.
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    # Method to draw a triangle on the frame to indicate certain positions.
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])  # Get the top y-coordinate of the bounding box.
        x, _ = get_center_of_bbox(bbox)  # Get the center x-coordinate of the bounding box.

        # Define points for the triangle.
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        # Draw the triangle on the frame.
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    # Method to display team ball control statistics on the frame.
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle for background.
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Calculate ball control statistics for both teams.
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        # Display ball control information on the frame.
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    # Method to draw annotations on all frames for players, referees, ball, and team ball control.
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []  # Initialize list to store annotated frames.

        # Loop through each frame and annotate it.
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Create a copy of the frame.

            # Retrieve track data for players, referees, and ball.
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players and their bounding boxes.
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default color is red.
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):  # If the player has the ball, draw a triangle.
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees' bounding boxes.
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Yellow color for referees.

            # Draw the ball's bounding box.
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))  # Green triangle for the ball.

            # Draw team ball control information.
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)  # Add the annotated frame to the output list.

        return output_video_frames  # Return the list of annotated frames.
