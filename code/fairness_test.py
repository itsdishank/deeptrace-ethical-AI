import argparse
import csv
import cv2
import torch
from spatial_cnn import SpatialModel
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--labels_file", required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpatialModel().to(device).eval()

results = {}
with open(args.labels_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename, label, group = row['filename'], int(row['label']), row['demographic']
        img = cv2.imread(f"sample_data/images/{filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        x = torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0)/255
        x = (x - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
        x = x.to(device)

        with torch.no_grad():
            pred = model(x)
        pred_label = 1 if pred.item() > 0.5 else 0
        if group not in results:
            results[group] = {'correct':0, 'total':0}
        results[group]['correct'] += (pred_label==label)
        results[group]['total'] += 1

for group, stats in results.items():
    acc = stats['correct'] / stats['total'] * 100
    print(f"{group}: {acc:.2f}% accuracy")
