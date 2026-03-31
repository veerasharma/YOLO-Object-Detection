from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, required=True, help='Path to input image')
args = parser.parse_args()

model = YOLO('yolov8n.pt')

results = model(args.source, save=True)

print("Detection completed!")
