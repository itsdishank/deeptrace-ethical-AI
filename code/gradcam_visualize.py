import argparse
import cv2
import torch
import numpy as np
from spatial_cnn import SpatialModel

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--output", default="gradcam_output.jpg")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpatialModel().to(device).eval()

orig = cv2.imread(args.image)
img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255
x = (x - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
x = x.to(device)

# Hook to capture gradients
activation = {}
def save_activation(module, input, output):
    activation['value'] = output
def save_gradient(module, grad_in, grad_out):
    activation['grad'] = grad_out[0]

model.base_model.features[-1].register_forward_hook(save_activation)
model.base_model.features[-1].register_backward_hook(save_gradient)

output = model.base_model(x)
score = output[0]
model.base_model.zero_grad()
score.backward()

grad = activation['grad']
act = activation['value']

weights = grad.mean(dim=[2,3], keepdim=True)
cam = (weights * act).sum(dim=1).squeeze()
cam = cam.cpu().detach().numpy()
cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

cv2.imwrite(args.output, overlay)
print(f"Grad-CAM saved to {args.output}")
