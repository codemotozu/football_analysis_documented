import sys  # Importing the `sys` module to manipulate the Python runtime environment.

# Adding the parent directory to the system path to allow importing utility functions from another folder.
sys.path.append('../')

# Importing utility functions:
# `get_center_of_bbox` calculates the center point of a bounding box (used to locate objects like players or the ball).
# `measure_distance` calculates the Euclidean distance between two points.
from utils import get_center_of_bbox, measure_distance

# Defining a class to assign the ball to the nearest player.
class PlayerBallAssigner():
    def __init__(self):
        # Maximum allowed distance between a player and the ball for an assignment to be made.
        self.max_player_ball_distance = 70

    # Function to assign the ball to the closest player.
    def assign_ball_to_player(self, players, ball_bbox):
        # Calculate the center position of the ball based on its bounding box (bbox).
        ball_position = get_center_of_bbox(ball_bbox)

        # Initialize variables to track the minimum distance and the assigned player's ID.
        miniumum_distance = 99999  # A very high initial value to ensure any actual distance is smaller.
        assigned_player = -1  # Default to -1, indicating no player is assigned initially.

        # Iterate over all detected players in the frame.
        for player_id, player in players.items():
            # Get the bounding box of the player.
            player_bbox = player['bbox']

            # Calculate the distances from the ball to the player's left and right edges.
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            
            # Determine the smaller of the two distances to represent the player's proximity to the ball.
            distance = min(distance_left, distance_right)

            # Check if this distance is below the maximum threshold for assigning the ball.
            if distance < self.max_player_ball_distance:
                # If the distance is also the smallest encountered so far, update the assigned player.
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id  # Assign the current player ID.

        # Return the ID of the player assigned to the ball, or -1 if no player meets the criteria.
        return assigned_player
