import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class HeuristicClassifier:
    """Fast gestures based purely on MediaPipe landmarks without CNN overhead."""
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def get_fingers_up(self, lmList):
        if len(lmList) == 0:
            return []
            
        fingers = []
        # Thumb (assuming right hand for simplicity, x coordinate check)
        if lmList[self.tip_ids[0]][1] > lmList[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers (y coordinate check against PIP joints)
        for id in range(1, 5):
            if lmList[self.tip_ids[id]][2] < lmList[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

class GestureCNN(nn.Module):
    """Lightweight CNN for complex gesture classification."""
    def __init__(self, num_classes=5):
        super(GestureCNN, self).__init__()
        # Input: 3x64x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 64 -> 32 -> 16. So 32 channels * 16 * 16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class HybridClassifier:
    """Wraps both heuristics and CNN for the pipeline"""
    def __init__(self, model_path=None, num_classes=5):
        self.heuristic = HeuristicClassifier()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn = GestureCNN(num_classes).to(self.device)
        if model_path:
            try:
                self.cnn.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load model {model_path}. Using untrained random weights. ({e})")
        self.cnn.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.classes = ['Fist', 'Pinch', 'Open', 'TwoFingers', 'ThumbUp']

    def infer_cnn(self, roi_img):
        # Check empty or invalid ROI
        if roi_img is None or roi_img.size == 0:
            return "None"
            
        img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.cnn(tensor)
            _, predicted = torch.max(outputs, 1)
        return self.classes[predicted.item()]
