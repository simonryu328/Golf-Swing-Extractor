import json
import numpy as np
from collections import defaultdict
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            frame_data = json.loads(line)
            frame_data['frame_number'] = i  # Add frame number
            data.append(frame_data)
    return data

def extract_trajectories(data):
    trajectories = defaultdict(lambda: defaultdict(list))
    image_size = None
    
    for frame in data:
        frame_number = frame['frame_number']
        if image_size is None:
            image_size = (frame['image']['width'], frame['image']['height'])
        
        # Initialize all object positions as NaN for this frame
        for obj_class in ['club', 'club_head', 'golfball', 'person']:
            trajectories[obj_class]['frame'].append(frame_number)
            trajectories[obj_class]['x'].append(float('nan'))
            trajectories[obj_class]['y'].append(float('nan'))
            trajectories[obj_class]['confidence'].append(float('nan'))
        
        # Update with actual detections
        for obj in frame['predictions']:
            obj_class = obj['class']
            if obj_class in ['club', 'club_head', 'golfball', 'person']:
                trajectories[obj_class]['x'][-1] = obj['x'] / image_size[0]  # Normalize
                trajectories[obj_class]['y'][-1] = obj['y'] / image_size[1]  # Normalize
                trajectories[obj_class]['confidence'][-1] = obj['confidence']
    
    return dict(trajectories), image_size

def interpolate_trajectories(trajectories):
    for obj_class, obj_data in trajectories.items():
        frames = np.array(obj_data['frame'])
        for coord in ['x', 'y']:
            values = np.array(obj_data[coord], dtype=float)
            mask = ~np.isnan(values) & (values != None)  # Handle both None and NaN
            if np.any(mask):
                valid_frames = frames[mask]
                valid_values = values[mask]
                interpolated = np.interp(frames, valid_frames, valid_values)
                trajectories[obj_class][coord] = interpolated.tolist()
    return trajectories

def create_sequences(trajectories, window_size=100, stride=10):
    sequences = []
    max_frames = max(len(trajectories[obj]['frame']) for obj in trajectories)
    
    for i in range(0, max_frames - window_size, stride):
        seq = {obj: {
            'x': trajectories[obj]['x'][i:i+window_size],
            'y': trajectories[obj]['y'][i:i+window_size],
            'confidence': trajectories[obj]['confidence'][i:i+window_size]
        } for obj in trajectories}
        seq['start_frame'] = i
        seq['end_frame'] = i + window_size
        sequences.append(seq)
    
    return sequences

def detect_swings(sequences):
    swing_labels = []
    for seq in sequences:
        # Check if club and club_head are present
        if 'club' in seq and 'club_head' in seq:
            club_velocity = np.diff([x for x in seq['club']['y'] if not np.isnan(x)])
            club_head_velocity = np.diff([x for x in seq['club_head']['y'] if not np.isnan(x)])
            
            # Simple heuristic: large velocity change in both club and club_head
            if len(club_velocity) > 0 and len(club_head_velocity) > 0:
                max_club_velocity = np.max(np.abs(club_velocity))
                max_club_head_velocity = np.max(np.abs(club_head_velocity))
                
                if max_club_velocity > 0.1 and max_club_head_velocity > 0.1:  # Threshold can be adjusted
                    swing_labels.append(1)
                else:
                    swing_labels.append(0)
            else:
                swing_labels.append(0)
        else:
            swing_labels.append(0)
    
    return swing_labels

class LabelingTool(QWidget):
    def __init__(self, sequences, initial_labels):
        super().__init__()
        self.sequences = sequences
        self.labels = initial_labels
        self.current_index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Swing Labeling Tool')
        layout = QVBoxLayout()

        self.label = QLabel(f"Sequence {self.current_index + 1}/{len(self.sequences)}")
        layout.addWidget(self.label)

        self.plot_widget = QWidget()
        layout.addWidget(self.plot_widget)

        self.swing_button = QPushButton('Swing')
        self.swing_button.clicked.connect(lambda: self.label_sequence(1))
        layout.addWidget(self.swing_button)

        self.no_swing_button = QPushButton('No Swing')
        self.no_swing_button.clicked.connect(lambda: self.label_sequence(0))
        layout.addWidget(self.no_swing_button)

        self.setLayout(layout)
        self.plot_sequence()

    def plot_sequence(self):
        plt.clf()
        seq = self.sequences[self.current_index]
        plt.plot(seq['club']['x'], seq['club']['y'], label='Club')
        plt.plot(seq['club_head']['x'], seq['club_head']['y'], label='Club Head')
        plt.plot(seq['golfball']['x'], seq['golfball']['y'], label='Golf Ball')
        plt.legend()
        plt.title(f"Sequence {self.current_index + 1}")
        plt.savefig('temp_plot.png')
        self.plot_widget.setStyleSheet("background-image: url(temp_plot.png);")

    def label_sequence(self, label):
        self.labels[self.current_index] = label
        self.current_index += 1
        if self.current_index < len(self.sequences):
            self.plot_sequence()
            self.label.setText(f"Sequence {self.current_index + 1}/{len(self.sequences)}")
        else:
            self.close()

# Main execution
if __name__ == "__main__":
    data = load_data('predictions/01_predictions.jsonl')
    trajectories, image_size = extract_trajectories(data)
    trajectories = interpolate_trajectories(trajectories)
    sequences = create_sequences(trajectories)
    initial_labels = detect_swings(sequences)

    # Print some statistics
    print(f"Total frames: {len(data)}")
    print(f"Image size: {image_size}")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Number of potential swings detected: {sum(initial_labels)}")

    # Here you would typically save the sequences and labels for further processing
    # For example:
    # with open('processed_data.json', 'w') as f:
    #     json.dump({'sequences': sequences, 'labels': initial_labels}, f)

    app = QApplication([])
    tool = LabelingTool(sequences, initial_labels)
    tool.show()
    app.exec_()

    final_labels = tool.labels
    # Now you have your sequences and verified labels
    # You can save these for further processing or model training