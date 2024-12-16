# Importing the YOLO class from the ultralytics library.
# This class provides methods to load and use pre-trained YOLO (You Only Look Once) models for object detection and tracking.
from ultralytics import YOLO 

# Loading a pre-trained YOLO model.
# The model weights are stored in the file 'models/best.pt'.
# This file contains the parameters of a trained YOLO network.
model = YOLO('models/best.pt')

# Using the model to perform predictions on a video file.
# The `predict` method analyzes the video ('input_videos/08fd33_4.mp4') frame by frame.
# The `save=True` argument saves the prediction results, including bounding boxes, to an output directory.
results = model.predict('input_videos/08fd33_4.mp4', save=True)

# Printing the results for the first frame of the video.
# This typically includes the detected objects, their bounding boxes, and associated metadata like confidence scores.
print(results[0])

# Printing a separator line for clarity in the console output.
print('=====================================')

# Iterating through the bounding boxes detected in the first frame.
# `results[0].boxes` contains a list of all the bounding boxes for objects detected in the frame.
for box in results[0].boxes:
    # Printing details of each bounding box, such as coordinates, confidence score, and class label.
    print(box)
