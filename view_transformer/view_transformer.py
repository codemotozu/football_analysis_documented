# Import required libraries
import numpy as np  # For numerical operations and array manipulation.
import cv2  # OpenCV library for computer vision tasks.

# Define a class for transforming points from pixel coordinates to real-world court coordinates.
class ViewTransformer():
    def __init__(self):
        # Define the real-world dimensions of the court in meters.
        court_width = 68  # Width of the court.
        court_length = 23.32  # Length of the court.

        # Define the pixel coordinates of the court vertices as seen in the video frame.
        # These represent key points on the court in the video.
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner in the video frame.
            [265, 275],   # Top-left corner in the video frame.
            [910, 260],   # Top-right corner in the video frame.
            [1640, 915]   # Bottom-right corner in the video frame.
        ])

        # Define the corresponding real-world coordinates for the court vertices in meters.
        # These are mapped to the actual physical dimensions of the court.
        self.target_vertices = np.array([
            [0, court_width],       # Bottom-left corner of the court in real-world coordinates.
            [0, 0],                 # Top-left corner of the court.
            [court_length, 0],      # Top-right corner of the court.
            [court_length, court_width]  # Bottom-right corner of the court.
        ])

        # Convert both the pixel and target vertices to float32 type for OpenCV processing.
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Create a perspective transformation matrix using OpenCV.
        # This matrix maps the pixel coordinates to the real-world court coordinates.
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    # Method to transform a single point from pixel space to real-world space.
    def transform_point(self, point):
        # Ensure the point is an integer tuple, as required for OpenCV functions.
        p = (int(point[0]), int(point[1]))

        # Check if the point is inside the polygon defined by the pixel vertices.
        # This ensures the point lies within the court boundaries.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None  # If the point is outside, return None.

        # Reshape the point to the required format for perspective transformation.
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        # Perform the perspective transformation to get the real-world coordinates.
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)

        # Reshape the transformed point to a flat array and return it.
        return tranform_point.reshape(-1, 2)

    # Method to add transformed positions to tracks.
    # This method updates the position of each tracked object to its real-world coordinates.
    def add_transformed_position_to_tracks(self, tracks):
        # Loop through all objects being tracked (e.g., players, ball).
        for object, object_tracks in tracks.items():
            # Loop through each frame of the object tracks.
            for frame_num, track in enumerate(object_tracks):
                # Loop through each track ID in the current frame.
                for track_id, track_info in track.items():
                    # Get the adjusted position (accounting for camera movement) of the object.
                    position = track_info['position_adjusted']

                    # Convert the position to a NumPy array.
                    position = np.array(position)

                    # Transform the position to real-world coordinates.
                    position_trasnformed = self.transform_point(position)

                    # If the transformation was successful, convert the transformed position to a list.
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()

                    # Update the tracks with the transformed position.
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
