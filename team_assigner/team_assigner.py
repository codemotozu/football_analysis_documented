# Import KMeans clustering from scikit-learn
from sklearn.cluster import KMeans

# Define a class for assigning players to teams based on their colors
class TeamAssigner:
    def __init__(self):
        # Dictionary to store team colors. Key: team ID, Value: average color of the team.
        self.team_colors = {}
        # Dictionary to map player IDs to their assigned teams.
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Reshape the image into a 2D array where each row represents a pixel (R, G, B values).
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters (representing 2 teams).
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)  # Fit the clustering model to the pixel data.

        # Return the fitted KMeans model.
        return kmeans

    def get_player_color(self, frame, bbox):
        # Extract the bounding box region of the player from the frame.
        # bbox = (x1, y1, x2, y2) format, specifying the player's location in the frame.
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Extract the top half of the player's bounding box (likely the jersey area).
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get a clustering model using K-means for the top half of the player's bounding box.
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel in the image (indicating which cluster they belong to).
        labels = kmeans.labels_

        # Reshape the labels array to match the shape of the original image.
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine which cluster corresponds to the player's jersey color.
        # Consider the corners of the bounding box (background pixels).
        corner_clusters = [
            clustered_image[0, 0],  # Top-left corner
            clustered_image[0, -1],  # Top-right corner
            clustered_image[-1, 0],  # Bottom-left corner
            clustered_image[-1, -1]  # Bottom-right corner
        ]
        # Identify the most common cluster in the corners as the non-player cluster (background).
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # The player's cluster is the other cluster.
        player_cluster = 1 - non_player_cluster

        # Retrieve the color of the player's cluster from the cluster centers.
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        # List to store the colors of all detected players.
        player_colors = []
        
        # Iterate through each player detection and extract their jersey color.
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]  # Get the player's bounding box.
            player_color = self.get_player_color(frame, bbox)  # Get the player's color.
            player_colors.append(player_color)  # Append the color to the list.
        
        # Use K-means to cluster the players' colors into 2 teams.
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)  # Fit the model to the collected player colors.

        # Store the fitted KMeans model for later use in predicting team assignments.
        self.kmeans = kmeans

        # Assign the cluster centers (average colors) to each team.
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # If the player's team has already been assigned, return it.
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Otherwise, determine the player's team based on their color.
        player_color = self.get_player_color(frame, player_bbox)  # Get the player's color.

        # Predict the team cluster for the player's color using the KMeans model.
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]

        # Adjust the team ID to be 1 or 2 (clusters are indexed at 0).
        team_id += 1

        # Special case: Manually assign team 1 to a specific player ID (e.g., referee).
        if player_id == 91:
            team_id = 1

        # Save the player's team assignment to the dictionary.
        self.player_team_dict[player_id] = team_id

        # Return the team ID.
        return team_id

