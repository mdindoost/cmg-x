import csv
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, filepath, header=None):
        self.filepath = filepath
        if header is not None and not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp'] + header)

    def log(self, values):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp] + values)
