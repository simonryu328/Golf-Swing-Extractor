import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtWebEngineWidgets import QWebEngineView
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrajectoryPlotter(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Trajectory Plotter")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        self.label_button = QPushButton("Label Sequence")
        self.label_button.clicked.connect(self.label_sequence)
        self.layout.addWidget(self.label_button)

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName()
        if file_name:
            with open(file_name, 'r') as file:
                data = [json.loads(line) for line in file]
            fig = self.create_trajectory_plot(data)
            self.web_view.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def create_trajectory_plot(self, data):
        # Your function remains the same
        all_predictions = []
        for frame_num, frame_data in enumerate(data):
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

    def label_sequence(self):
        # Implement labeling logic here
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrajectoryPlotter()
    window.show()
    sys.exit(app.exec_())