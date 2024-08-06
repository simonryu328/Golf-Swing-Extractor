import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from ipywidgets import interactive, IntSlider, Button, Output, VBox, HBox, Text, Layout
from IPython.display import display
import random

class GolfSwingAnalyzer:
    def __init__(self, video_file, video_dir='input_videos', predictions_dir='predictions'):
        self.video_file = video_file
        self.video_path = os.path.join(video_dir, video_file)
        self.predictions_path = os.path.join(predictions_dir, f"{os.path.splitext(video_file)[0]}_predictions.jsonl")
        self.data = self.load_data_from_jsonl()
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
        self.swing_intervals = []
        self.current_interval = [None, None]

    def load_data_from_jsonl(self):
        data = []
        with open(self.predictions_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def load_swing_intervals(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def process_video(self, swing_intervals_file=None, sequence_length=64, overlap=32, swing_threshold=0.7):
        predictions = self.data
        if swing_intervals_file:
            swing_intervals = self.load_swing_intervals(swing_intervals_file)
        else:
            swing_intervals = None
        
        # Extract image dimensions from the first prediction
        img_width = predictions[0]['image']['width']
        img_height = predictions[0]['image']['height']
        
        # Process predictions
        processed_frames = {}
        for frame, pred in enumerate(predictions):
            frame_data = {}
            for p in pred['predictions']:
                if p['class'] in ['club', 'club_head']:
                    frame_data[p['class']] = [
                        p['x'] / img_width,
                        p['y'] / img_height
                    ]
            
            # Only add frame data if both club and club_head are detected
            if 'club' in frame_data and 'club_head' in frame_data:
                processed_frames[frame] = [
                    *frame_data['club'],
                    *frame_data['club_head']
                ]
        
        # Create sequences
        sequences = []
        labels = []
        frames = []
        for i in range(0, len(processed_frames), overlap):
            seq_frames = list(processed_frames.keys())[i:i+sequence_length]
            # Add the first frame of the sequence to the frames list
            frames.append(seq_frames[0])
            seq = [processed_frames[j] for j in seq_frames if j in processed_frames]
            
            # Check if sequence is complete
            if len(seq) == sequence_length:
                sequences.append(seq)
                
                # Label the sequence if swing intervals are provided
                if swing_intervals:
                    swing_frames = [frame for frame in seq_frames if any(start <= frame < end for start, end in swing_intervals)]
                    labels.append(1 if len(swing_frames) / len(seq_frames) > swing_threshold else 0)
        
        return np.array(sequences), np.array(labels), frames

    def plot_sample_sequences(self, sequences, labels, num_samples=3):
        # Get indices of positive and negative samples
        positive_indices = np.where(labels == 1)[0]
        negative_indices = np.where(labels == 0)[0]
        
        # Randomly sample from positive and negative sequences
        sample_positive = random.sample(list(positive_indices), min(num_samples, len(positive_indices)))
        sample_negative = random.sample(list(negative_indices), min(num_samples, len(negative_indices)))
        
        # Create subplots: 2 rows (positive/negative) x num_samples columns x 2 sub-rows (x/y coordinates)
        fig = make_subplots(rows=4, cols=num_samples, 
                            subplot_titles=(['Positive Samples']*num_samples + ['Negative Samples']*num_samples),
                            vertical_spacing=0.1,
                            row_heights=[0.23, 0.23, 0.23, 0.23])
        
        def plot_sequence(seq, start_row, col):
            frames = np.arange(len(seq))
            
            # Plot x coordinates
            fig.add_trace(go.Scatter(x=frames, y=seq[:, 0], mode='lines+markers', name='Club X', 
                                     line=dict(color='blue'), showlegend=start_row==1 and col==1), 
                          row=start_row, col=col)
            fig.add_trace(go.Scatter(x=frames, y=seq[:, 2], mode='lines+markers', name='Club Head X', 
                                     line=dict(color='red'), showlegend=start_row==1 and col==1), 
                          row=start_row, col=col)
            
            # Plot y coordinates
            fig.add_trace(go.Scatter(x=frames, y=seq[:, 1], mode='lines+markers', name='Club Y', 
                                     line=dict(color='blue'), showlegend=False), 
                          row=start_row+1, col=col)
            fig.add_trace(go.Scatter(x=frames, y=seq[:, 3], mode='lines+markers', name='Club Head Y', 
                                     line=dict(color='red'), showlegend=False), 
                          row=start_row+1, col=col)
        
        # Plot positive samples
        for i, idx in enumerate(sample_positive):
            plot_sequence(sequences[idx], 1, i+1)
        
        # Plot negative samples
        for i, idx in enumerate(sample_negative):
            plot_sequence(sequences[idx], 3, i+1)
        
        # Update layout
        fig.update_layout(height=1200, width=1200, title_text="Sample Sequences: Positive vs Negative")
        fig.update_xaxes(title_text="Frame Number")
        fig.update_yaxes(title_text="Normalized Coordinate", range=[0, 1])
        
        # Add y-axis titles
        for i in range(1, num_samples + 1):
            fig.update_yaxes(title_text="X Coordinate", row=1, col=i)
            fig.update_yaxes(title_text="Y Coordinate", row=2, col=i)
            fig.update_yaxes(title_text="X Coordinate", row=3, col=i)
            fig.update_yaxes(title_text="Y Coordinate", row=4, col=i)
        
        # Show the plot
        fig.show()

    def create_trajectory_plot(self, start_frame=None, end_frame=None):
        all_predictions = []
        for frame_num, frame_data in enumerate(self.data):
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

        start_frame = 0 if start_frame is None else start_frame
        end_frame = self.total_frames - 1 if end_frame is None else end_frame

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
        trajectory_fig = self.create_trajectory_plot(start_frame, end_frame)
        frames_fig = self.plot_evenly_distributed_frames(start_frame, end_frame, num_frames)

        trajectory_fig.show()
        plt.figure(frames_fig.number)
        plt.show()

    def save_swing_intervals(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{os.path.splitext(self.video_file)[0]}_swing_intervals.json")
        with open(output_file, 'w') as f:
            json.dump(self.swing_intervals, f)
        print(f"Swing intervals saved to {output_file}")

    def find_swing_intervals(self):
        trajectory_fig = self.create_trajectory_plot()
        trajectory_fig.update_layout(height=600)

        image_output = Output()
        trajectory_output = Output()
        
        with trajectory_output:
            display(trajectory_fig)
        
        def update_plot(frame):
            with image_output:
                image_output.clear_output(wait=True)
                
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, img = cap.read()
                cap.release()
                
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.title(f"Frame {frame}")
                    plt.axis('off')
                    plt.show()

        def mark_start(b):
            self.current_interval[0] = frame_slider.value
            print(f"Marked start of swing at frame {self.current_interval[0]}")

        def mark_end(b):
            self.current_interval[1] = frame_slider.value
            if self.current_interval[0] is not None and self.current_interval[1] > self.current_interval[0]:
                self.swing_intervals.append(tuple(self.current_interval))
                print(f"Added swing interval: {self.current_interval}")
                self.current_interval[0] = None
                self.current_interval[1] = None
            else:
                print("Invalid interval. Make sure to mark start before end and end frame is after start frame.")

        def show_intervals(b):
            print("Current swing intervals:")
            for interval in self.swing_intervals:
                print(interval)

        def check_intervals(b):
            valid = True
            for interval in self.swing_intervals:
                start, end = interval
                if not (end > start and (end - start) <= 80):
                    valid = False
                    print(f"Invalid interval: {interval}")
            if valid:
                print("All intervals are valid.")

        def set_frame(b):
            try:
                frame = int(frame_input.value)
                if 0 <= frame < self.total_frames:
                    frame_slider.value = frame
                else:
                    print(f"Frame number must be between 0 and {self.total_frames - 1}")
            except ValueError:
                print("Please enter a valid integer for the frame number")

        def increment_frame(b):
            if frame_slider.value < self.total_frames - 1:
                frame_slider.value += 1

        def decrement_frame(b):
            if frame_slider.value > 0:
                frame_slider.value -= 1

        def save_intervals(b):
            self.save_swing_intervals('swing_intervals')

        frame_slider = IntSlider(min=0, max=self.total_frames-1, step=1, description='Frame:', layout=Layout(width='800px'))
        frame_input = Text(description='Go to frame:')
        set_frame_button = Button(description="Set Frame")
        increment_frame_button = Button(description="Next Frame")
        decrement_frame_button = Button(description="Previous Frame")
        start_button = Button(description="Mark Start")
        end_button = Button(description="Mark End")
        show_button = Button(description="Show Intervals")
        save_button = Button(description="Save Intervals")
        check_button = Button(description="Check Intervals")

        start_button.on_click(mark_start)
        end_button.on_click(mark_end)
        show_button.on_click(show_intervals)
        set_frame_button.on_click(set_frame)
        increment_frame_button.on_click(increment_frame)
        decrement_frame_button.on_click(decrement_frame)
        save_button.on_click(save_intervals)
        check_button.on_click(check_intervals)

        interactive_plot = interactive(update_plot, frame=frame_slider)
        
        display(VBox([
            trajectory_output,
            interactive_plot,
            image_output,
            HBox([frame_input, set_frame_button]),
            HBox([decrement_frame_button, increment_frame_button]),
            HBox([start_button, end_button, show_button, check_button, save_button])
        ]))

# Usage example:
# analyzer = GolfSwingAnalyzer("your_video_file.mp4")
# analyzer.combined_plot()
# analyzer.find_swing_intervals()