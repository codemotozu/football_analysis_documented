import cv2  # OpenCV library for computer vision tasks, used for drawing and text on frames.
import sys  # Provides access to system-specific parameters and functions.

# Add the parent directory to the system path to allow importing utility modules from it.
sys.path.append('../')

# Import utility functions for measuring distance and determining foot positions.
from utils import measure_distance, get_foot_position

# Class for estimating speed and distance covered by objects (e.g., players).
class SpeedAndDistance_Estimator():
    def __init__(self):
        # Define a frame window for calculating speed and distance over a batch of frames.
        self.frame_window = 5
        # Frame rate of the video in frames per second (FPS), used for time calculations.
        self.frame_rate = 24

    # Method to add speed and distance data to tracking information.
    def add_speed_and_distance_to_tracks(self, tracks):
        # Dictionary to store the total distance covered by each tracked object.
        total_distance = {}

        # Iterate over all tracked objects (e.g., players, referees, ball).
        for object, object_tracks in tracks.items():
            # Skip objects of type "ball" or "referees" as they are not relevant for this calculation.
            if object == "ball" or object == "referees":
                continue
            
            # Get the total number of frames in the track.
            number_of_frames = len(object_tracks)
            
            # Process frames in batches of size `frame_window`.
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Determine the last frame in the current batch (or the last available frame).
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # Iterate over all track IDs in the current frame.
                for track_id, _ in object_tracks[frame_num].items():
                    # Skip if the track ID is not present in the last frame of the batch.
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the starting and ending transformed positions of the object.
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if either position is None (e.g., due to missing data).
                    if start_position is None or end_position is None:
                        continue

                    # Calculate the distance covered by the object in this batch.
                    distance_covered = measure_distance(start_position, end_position)
                    
                    # Calculate the time elapsed during the batch in seconds.
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    
                    # Calculate the speed in meters per second.
                    speed_meteres_per_second = distance_covered / time_elapsed
                    
                    # Convert speed to kilometers per hour.
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    # Initialize the total distance for the object type if not already present.
                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    # Initialize the total distance for the specific track ID if not already present.
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    # Accumulate the distance covered in this batch to the total distance.
                    total_distance[object][track_id] += distance_covered

                    # Add speed and distance information to each frame in the current batch.
                    for frame_num_batch in range(frame_num, last_frame):
                        # Skip if the track ID is not present in the current frame.
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        
                        # Store the speed and cumulative distance for the track ID in the current frame.
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    # Method to draw speed and distance information on video frames.
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []  # List to store the annotated frames.

        # Iterate over each frame and its corresponding tracking data.
        for frame_num, frame in enumerate(frames):
            # Process tracking data for all object types in the current frame.
            for object, object_tracks in tracks.items():
                # Skip objects of type "ball" or "referees" as they are not displayed.
                if object == "ball" or object == "referees":
                    continue
                
                # Iterate over each track in the current frame.
                for _, track_info in object_tracks[frame_num].items():
                    # Check if speed and distance information is available for the track.
                    if "speed" in track_info:
                        # Retrieve the speed and distance values.
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)

                        # Skip if either value is missing.
                        if speed is None or distance is None:
                            continue

                        # Get the bounding box of the object.
                        bbox = track_info['bbox']
                        
                        # Determine the position for displaying the text (near the object's feet).
                        position = get_foot_position(bbox)  # Calculate the foot position from the bounding box.
                        position = list(position)  # Convert the position to a list for manipulation.
                        position[1] += 40  # Offset the text position slightly below the object.

                        # Convert the position back to a tuple of integers for OpenCV.
                        position = tuple(map(int, position))
                        
                        # Draw the speed on the frame near the object.
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        # Draw the cumulative distance on the frame below the speed text.
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add the annotated frame to the output list.
            output_frames.append(frame)
        
        # Return the list of frames with annotations.
        return output_frames
