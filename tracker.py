import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import os
from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class FootballTracker:
    def __init__(self):
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.device)
        self.track_history = defaultdict(list)
        self.player_stats = defaultdict(lambda: {
            'distance': 0,
            'frames': 0,
            'speed': 0,
            'team': None
        })
        self.frame_count = 0
        self.MAX_PLAYERS = 30
        self.player_colors = {}
        self.team_colors = {}
        
    def assign_team_colors(self, frame, boxes, track_ids):
        """Cluster jersey colors to identify teams and players"""
        if len(boxes) < 2:
            return
            
        # Extract jersey patches
        patches = []
        for box in boxes:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            patch = frame[max(0, y-h//4):y+h//4, max(0, x-w//4):x+w//4]
            if patch.size > 0:
                patches.append(patch)
        
        if not patches:
            return
            
        # Get dominant colors
        dominant_colors = []
        for patch in patches:
            pixels = patch.reshape(-1, 3)
            if len(pixels) > 10:  # Ensure enough pixels for clustering
                kmeans = KMeans(n_clusters=1, n_init=3)
                kmeans.fit(pixels)
                dominant_colors.append(kmeans.cluster_centers_[0])
        
        # Cluster into two teams and assign labels to players
        if len(dominant_colors) > 1:
            kmeans = KMeans(n_clusters=2, n_init=3)
            kmeans.fit(dominant_colors)

            self.team_colors = {
                0: (255, 0, 0),  # Team A - Blue
                1: (0, 0, 255)   # Team B - Red
            }

            for track_id, label in zip(track_ids, kmeans.labels_):
                self.player_stats[track_id]['team'] = int(label)
    
    def get_player_color(self, track_id):
        """Assign unique color to each player"""
        if track_id not in self.player_colors:
            # Generate distinct color based on ID
            hue = (track_id * 73) % 181  # Prime number for dispersion
            # FIXED: Correct bracket structure
            hsv_color = np.uint8([[[hue, 255, 220]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
            color = bgr_color[0][0]
            self.player_colors[track_id] = (int(color[0]), int(color[1]), int(color[2]))
        return self.player_colors[track_id]
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def process_frame(self, frame):
        try:
            # Downsample frame
            original_frame = frame.copy()
            frame = cv2.resize(frame, (640, 360))
            
            # Run tracking
            results = self.model.track(
                frame,
                device=self.device,
                half=True,
                verbose=False,
                tracker="bytetrack.yaml",
                conf=0.5
            )
            
            if not results or results[0].boxes.id is None:
                return original_frame, self.player_stats
                
            # Extract tracking data
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Assign teams on first frame
            if self.frame_count == 0:
                self.assign_team_colors(original_frame, boxes, track_ids)
            
            # Process each player
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                current_point = (float(x), float(y))
                
                # Update track history
                if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                    last_point = self.track_history[track_id][-1]
                    distance = self.calculate_distance(last_point, current_point)
                    self.player_stats[track_id]['distance'] += distance
                    
                    # Update speed (pixels/frame)
                    self.player_stats[track_id]['speed'] = 0.8 * self.player_stats[track_id].get('speed', 0) + 0.2 * distance
                
                self.track_history[track_id].append(current_point)
                self.player_stats[track_id]['frames'] += 1
                
                # Limit history length
                if len(self.track_history[track_id]) > 100:
                    self.track_history[track_id].pop(0)
            
            # Prune inactive tracks
            active_ids = set(track_ids)
            self.track_history = {k: v for k, v in self.track_history.items() if k in active_ids or len(v) > 0}
            self.track_history = dict(sorted(
                self.track_history.items(),
                key=lambda item: len(item[1]),
                reverse=True
            )[:self.MAX_PLAYERS])
            
            # Visualize tracks
            annotated_frame = results[0].plot()
            
            # Draw player trajectories
            for track_id, trail in self.track_history.items():
                if len(trail) > 1:
                    color = self.get_player_color(track_id)
                    team = self.player_stats[track_id].get('team', 0)
                    team_color = self.team_colors.get(team, (0, 255, 0))
                    
                    # Draw trail
                    for i in range(1, len(trail)):
                        cv2.line(
                            annotated_frame,
                            (int(trail[i-1][0]), int(trail[i-1][1])),
                            (int(trail[i][0]), int(trail[i][1])),
                            color,
                            2
                        )
                    
                    # Draw current position with team color
                    last_point = trail[-1]
                    cv2.circle(annotated_frame, (int(last_point[0]), int(last_point[1])), 10, team_color, -1)
                    cv2.putText(
                        annotated_frame, 
                        str(track_id), 
                        (int(last_point[0]) - 10, int(last_point[1]) + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
            
            self.frame_count += 1
            return cv2.resize(annotated_frame, (original_frame.shape[1], original_frame.shape[0])), self.player_stats
            
        except Exception as e:
            print(f"Tracking error: {e}")
            return frame, self.player_stats
    
    def reset(self):
        self.track_history.clear()
        self.player_stats.clear()
        self.player_colors = {}
        self.team_colors = {}
        self.frame_count = 0
