import os
import json

def count_intervals(directory):
    total_intervals = 0

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                intervals = json.load(file)
                num_intervals = len(intervals)
                
                print(f"Number of intervals in {filename}: {num_intervals}")
                total_intervals += num_intervals

    print(f"Total number of intervals in all files: {total_intervals}")

# Specify the directory containing the JSON files
directory = "swing_intervals"

# Run the function
count_intervals(directory)
