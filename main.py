from utils import read_video, save_video  # Utility functions for reading and saving video files.
from trackers import Tracker  # Custom tracker class for object tracking.
import cv2  # OpenCV for video processing and visualization.
import numpy as np  # NumPy for numerical computations and array handling.
from team_assigner import TeamAssigner  # Module to assign teams to players.
from player_ball_assigner import PlayerBallAssigner  # Module to assign ball possession to players.
from camera_movement_estimator import CameraMovementEstimator  # Module to estimate camera movement.
from view_transformer import ViewTransformer  # Module to transform views for tracking consistency.
from speed_and_distance_estimator import SpeedAndDistance_Estimator  # Module to calculate speed and distance.

def main():
    # Step 1: Read video frames from a file.
    video_frames = read_video('input_videos/08fd33_4.mp4')  # Reads input video and returns a list of frames.

    # Step 2: Initialize the tracker with a pre-trained model for object detection and tracking.
    tracker = Tracker('models/best.pt')  # Loads a YOLO-based model for tracking objects in the video.

    # Step 3: Get object tracks for the video frames.
    # Optionally, read pre-computed tracks from a stub file for faster processing.
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True, 
                                       stub_path='stubs/track_stubs.pkl')

    # Step 4: Add positions to each object's track (computed positions per frame).
    tracker.add_position_to_tracks(tracks)

    # Step 5: Initialize the camera movement estimator using the first video frame.
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # Step 6: Estimate camera movement across all frames.
    # Optionally, load pre-computed data from a stub file.
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, 
                                                                              read_from_stub=True, 
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    # Step 7: Adjust object positions in the tracks based on camera movement.
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 8: Initialize the view transformer to normalize positions into a consistent view.
    view_transformer = ViewTransformer()

    # Step 9: Transform object positions in the tracks into the new view space.
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 10: Interpolate ball positions to fill in missing data (e.g., when the ball is occluded).
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 11: Calculate speed and distance for all tracked objects.
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 12: Assign teams to players based on their appearance in the first frame.
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Step 13: Iterate through each frame and assign teams to players based on bounding boxes.
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            # Determine the player's team and team color.
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team  # Store the team assignment.
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  # Assign color.

    # Step 14: Assign ball possession to players and determine which team controls the ball.
    player_assigner = PlayerBallAssigner()
    team_ball_control = []  # List to store which team controls the ball in each frame.
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Get the ball's bounding box for the frame.
        # Assign the ball to the nearest player.
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:  # If a player is assigned to the ball:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True  # Mark the player as having the ball.
            # Record which team the player belongs to for ball control.
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # If no player is assigned, use the previous frame's ball control information.
            team_ball_control.append(team_ball_control[-1])
    # Convert ball control list to a NumPy array for further analysis.
    team_ball_control = np.array(team_ball_control)

    # Step 15: Visualize results by drawing annotations on the video frames.
    ## Draw object tracks, including players and the ball.
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement annotations on the video frames.
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    ## Draw speed and distance annotations for all tracked objects.
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Step 16: Save the annotated video to an output file.
    save_video(output_video_frames, 'output_videos/output_video.avi')

# Execute the main function if this script is run directly.
if __name__ == '__main__':
    main()
