# Function to calculate the center point of a bounding box (bbox).
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox  # Unpacking the coordinates of the bounding box (top-left and bottom-right).
    # Calculate the center by averaging the x and y coordinates of the top-left and bottom-right corners.
    return int((x1 + x2) / 2), int((y1 + y2) / 2)  # Return the center as an integer tuple.

# Function to calculate the width of a bounding box.
def get_bbox_width(bbox):
    # The width is the horizontal distance between the left (x1) and right (x2) sides of the bbox.
    return bbox[2] - bbox[0]  # Return the width (x2 - x1).

# Function to measure the Euclidean distance between two points (p1, p2).
def measure_distance(p1, p2):
    # Calculate the square of the differences in x and y coordinates, then take the square root to get the distance.
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5  # Return the distance as a float.

# Function to measure the distance in the x and y directions separately (x distance and y distance).
def measure_xy_distance(p1, p2):
    # The difference in x coordinates and y coordinates separately.
    return p1[0] - p2[0], p1[1] - p2[1]  # Return the distance in x and y directions as a tuple.

# Function to get the foot position of a bounding box, assuming the bbox represents a person.
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox  # Unpacking the coordinates of the bounding box (top-left and bottom-right).
    # The foot position is typically assumed to be at the bottom center of the bounding box (y2 is the bottom).
    return int((x1 + x2) / 2), int(y2)  # Return the foot position as an integer tuple (x-center, y-bottom).
