import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from capture import VideoCaptureAsync
from tracker import HandTracker
from classifier import GestureCNN

class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Fist', 'Pinch', 'Open', 'TwoFingers', 'ThumbUp']
        self.filepaths = []
        self.labels = []
        
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.filepaths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)
                    
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def collect_data(num_samples=100, class_name='Fist', root='data'):
    cls_dir = os.path.join(root, class_name)
    os.makedirs(cls_dir, exist_ok=True)
    
    cap = VideoCaptureAsync(src=0).start()
    tracker = HandTracker(detection_con=0.8, track_con=0.8)
    
    print(f"Collecting {num_samples} samples for {class_name} in 3 seconds...")
    time.sleep(3)
    
    count = 0
    try:
        while count < num_samples:
            grabbed, img = cap.read()
            if not grabbed or img is None:
                time.sleep(0.01)
                continue
                
            img = tracker.find_hands(img, draw=False)
            lmList, bbox = tracker.get_position_and_bbox(img, draw=False)
            
            if len(lmList) != 0 and len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
                # Add padding
                h, w, c = img.shape
                pad = 20
                xmin = max(0, xmin - pad)
                ymin = max(0, ymin - pad)
                xmax = min(w, xmax + pad)
                ymax = min(h, ymax + pad)
                
                roi = img[ymin:ymax, xmin:xmax]
                if roi.size > 0:
                    cv2.imwrite(os.path.join(cls_dir, f"{int(time.time()*1000)}.jpg"), roi)
                    count += 1
                    
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(img, f'{count}/{num_samples}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            
            cv2.imshow("Data Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()
    print("Done collecting.")

def train_model(data_dir='data', epochs=10, batch_size=32, save_path='gesture_cnn.pth'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = HandGestureDataset(root_dir=data_dir, transform=transform)
    if len(dataset) == 0:
        print("No data found! Please collect data first.")
        return
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device}...")
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    # Usage example
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        cls = sys.argv[2] if len(sys.argv) > 2 else 'Fist'
        collect_data(num_samples=200, class_name=cls)
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_model(epochs=20)
    else:
        print("Usage: python train_cnn.py collect <ClassName> OR python train_cnn.py train")
