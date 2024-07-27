import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

class GolfSwingPlotter:
    def __init__(self, video_file, video_dir='input_videos', predictions_dir='predictions'):
        self.video_file = video_file
        self.video_path = os.path.join(video_dir, video_file)
        self.jsonl_path = os.path.join(predictions_dir, f"{os.path.splitext(video_file)[0]}_predictions.jsonl")

    def load_data_from_jsonl(self):
        data = []
        with open(self.jsonl_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def create_trajectory_plot(self, data, start_frame=None, end_frame=None):
        all_predictions = []
        for frame_num, frame_data in enumerate(data):
            if (start_frame is None or frame_num >= start_frame) and (end_frame is None or frame_num <= end_frame):
                for pred in frame_data.get('predictions', []):
                    pred['frame'] = frame_num
                    all_predictions.append(pred)
        
        df = pd.DataFrame(all_predictions)
        classes = df['class'].unique()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            subplot_titles=("X-coordinate Trajectory", "Y-coordinate Trajectory"))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        color_map = dict(zip(classes, colors[:len(classes)]))
        
        for cls in classes:
            class_data = df[df['class'] == cls].sort_values('frame')
            
            fig.add_trace(
                go.Scatter(x=class_data['frame'], y=class_data['x'], mode='lines+markers',
                           name=f'{cls} (x)', line=dict(color=color_map[cls])),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=class_data['frame'], y=class_data['y'], mode='lines+markers',
                           name=f'{cls} (y)', line=dict(color=color_map[cls], dash='dash')),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Object Trajectories in Golf Swing Video")
        fig.update_xaxes(title_text="Frame Number")
        fig.update_yaxes(title_text="X Position", row=1, col=1)
        fig.update_yaxes(title_text="Y Position", row=2, col=1)
        
        return fig

    def plot_evenly_distributed_frames(self, start_frame=None, end_frame=None, num_frames=9):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = 0 if start_frame is None else start_frame
        end_frame = total_frames - 1 if end_frame is None else end_frame

        frame_positions = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        cols = math.ceil(math.sqrt(num_frames))
        rows = math.ceil(num_frames / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, frame_position in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Error: Could not read frame at position {frame_position}.")
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {frame_position}')
            axes[i].axis('off')
        
        for j in range(len(frame_positions), len(axes)):
            axes[j].axis('off')
        
        cap.release()
        plt.tight_layout(pad=2.0)
        return fig

    def combined_plot(self, start_frame=None, end_frame=None, num_frames=9):
        data = self.load_data_from_jsonl()
        trajectory_fig = self.create_trajectory_plot(data, start_frame, end_frame)
        frames_fig = self.plot_evenly_distributed_frames(start_frame, end_frame, num_frames)

        trajectory_fig.show()
        plt.figure(frames_fig.number)
        plt.show()