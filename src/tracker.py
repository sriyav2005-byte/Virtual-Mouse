import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        self.max_hands = max_hands
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con
        )
            
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # EMA smoothing for the pointer
        self.prev_x, self.prev_y = 0, 0
        self.alpha_ema = 0.5
        self.results = None
        
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        self.results = self.detector.detect(mp_image)
        
        if self.results.hand_landmarks and draw:
            for hand_lms in self.results.hand_landmarks:
                h, w, c = img.shape
                points = []
                for lm in hand_lms:
                    points.append((int(lm.x * w), int(lm.y * h)))
                for p1, p2 in HAND_CONNECTIONS:
                    cv2.line(img, points[p1], points[p2], (0, 255, 0), 2)
                for p in points:
                    cv2.circle(img, p, 4, (0, 0, 255), cv2.FILLED)
                    
        return img
        
    def get_position_and_bbox(self, img, hand_no=0, draw=True):
        xList = []
        yList = []
        bbox = []
        lmList = []
        
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                myHand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                
                for id, lm in enumerate(myHand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    lmList.append([id, cx, cy])
                    
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                
                if draw:
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)
                              
        return lmList, bbox
        
    def smooth_point(self, curr_x, curr_y):
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = curr_x, curr_y
            return curr_x, curr_y
            
        smooth_x = int(self.alpha_ema * curr_x + (1 - self.alpha_ema) * self.prev_x)
        smooth_y = int(self.alpha_ema * curr_y + (1 - self.alpha_ema) * self.prev_y)
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
