import argparse
import cv2
import numpy as np
import torch
from spatial_cnn import SpatialModel
from temporal_lstm import TemporalModel

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    return (img - mean) / std

parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="Path to image or video")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spatial_model = SpatialModel().to(device).eval()
temporal_model = TemporalModel().to(device).eval()

if args.input_path.endswith(('.jpg', '.png', '.jpeg')):
    frame = cv2.imread(args.input_path)
    x = preprocess_frame(frame).to(device)
    with torch.no_grad():
        pred = spatial_model(x)
    print(f"Prediction: {'Fake' if pred.item() > 0.5 else 'Real'} (Confidence {pred.item()*100:.2f}%)")
else:
    cap = cv2.VideoCapture(args.input_path)
    frames = []
    spatial_preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frames) % 5 == 0:  # Sample every 5th frame
            x = preprocess_frame(frame).to(device)
            with torch.no_grad():
                pred = spatial_model(x)
            spatial_preds.append(pred.item())
            feat = spatial_model.base_model.features(x)
            pooled_feat = spatial_model.base_model.avgpool(feat)
            flat_feat = torch.flatten(pooled_feat, 1)
            frames.append(flat_feat.cpu())
    cap.release()
    if frames:
        seq = torch.cat(frames, dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            temp_pred = temporal_model(seq)
        spatial_avg = np.mean(spatial_preds)
        final_score = (spatial_avg + temp_pred.item()) / 2
        print(f"Video Prediction: {'Fake' if final_score > 0.5 else 'Real'} (Confidence {final_score*100:.2f}%)")
    else:
        print("No frames read from video.")
