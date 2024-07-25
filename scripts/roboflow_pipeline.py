import os
import json
from pathlib import Path
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GolfSwingDataPipeline:
    def __init__(self, input_dir, output_dir, model_id, api_key):
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.model_id = model_id
        self.api_key = api_key
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_video(self, video_path):
        video_name = video_path.stem
        output_file = self.output_dir / f"{video_name}_predictions.jsonl"
        
        predictions = []

        def save_prediction(prediction: dict, video_frame: VideoFrame) -> None:
            predictions.append(prediction)

        pipeline = InferencePipeline.init(
            model_id=self.model_id,
            video_reference=str(video_path),
            on_prediction=save_prediction,
            api_key=self.api_key,
        )

        logger.info(f"Processing video: {video_name}")
        pipeline.start()
        pipeline.join()

        with output_file.open('w') as f:
            for pred in predictions:
                json.dump(pred, f)
                f.write('\n')

        logger.info(f"Saved predictions for {video_name} to {output_file}")

    def run(self):
        video_files = list(self.input_dir.glob('*.mp4'))
        
        with ThreadPoolExecutor() as executor:
            executor.map(self.process_video, video_files)

if __name__ == "__main__":
    # Get the directory of the current script
    SCRIPT_DIR = Path(__file__).resolve().parent

    # Define paths relative to the script directory
    INPUT_DIR = SCRIPT_DIR.parent / "input_videos"
    OUTPUT_DIR = SCRIPT_DIR / "predictions"
    MODEL_ID = "golf-49wbh/1"
    API_KEY = os.getenv("ROBOFLOW_API_KEY")

    pipeline = GolfSwingDataPipeline(INPUT_DIR, OUTPUT_DIR, MODEL_ID, API_KEY)
    pipeline.run()