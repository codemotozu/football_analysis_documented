import cv2  # Import the OpenCV library for video processing and computer vision tasks.

# Function to read video frames from a given video file path.
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)  # Initialize the video capture object to open the video file.
    frames = []  # Create an empty list to store the frames of the video.
    
    while True:
        ret, frame = cap.read()  # Read a frame from the video. 'ret' is a boolean that indicates if the frame was successfully read.
        if not ret:  # If ret is False, the video has ended or there was an error reading a frame.
            break  # Exit the loop if no more frames are available.
        frames.append(frame)  # Append the successfully read frame to the frames list.
    
    return frames  # Return the list of all frames read from the video.

# Function to save video frames to a new video file.
def save_video(output_video_frames, output_video_path):
    # Create a VideoWriter object to save the video to a file.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Specify the codec to use for encoding the video (XVID here).
    # Initialize the VideoWriter with the desired output path, codec, frame rate (24 fps), and frame size (width x height).
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # Loop through all the frames in the provided list.
    for frame in output_video_frames:
        out.write(frame)  # Write the current frame to the output video.
    
    out.release()  # Release the VideoWriter object once all frames are written to the file.

