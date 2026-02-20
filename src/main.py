import cv2
import numpy as np
import time
from capture import VideoCaptureAsync
from tracker import HandTracker
from classifier import HybridClassifier
from controller import MouseController

def main():
    wCam, hCam = 640, 480
    frameR = 100 # Active area margin
    
    cap = VideoCaptureAsync(src=0, width=wCam, height=hCam).start()
    tracker = HandTracker(max_hands=1, detection_con=0.8, track_con=0.8)
    classifier = HybridClassifier(model_path='gesture_cnn.pth')
    mouse = MouseController()
    
    pTime = 0
    
    try:
        while True:
            grabbed, img = cap.read()
            if not grabbed or img is None:
                time.sleep(0.01)
                continue
                
            img = cv2.flip(img, 1) # Mirror display
            
            img = tracker.find_hands(img)
            lmList, bbox = tracker.get_position_and_bbox(img)
            
            # Draw frame ROI
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
            
            if len(lmList) != 0:
                # 1. Pointer coordinates (Index finger tip: 8)
                x1, y1 = lmList[8][1], lmList[8][2]
                
                # 2. Heuristic check
                fingers = classifier.heuristic.get_fingers_up(lmList)
                
                if len(fingers) == 5:
                    # Move: index finger only
                    if fingers[1] == 1 and fingers[2] == 0:
                        mx, my = mouse.get_screen_coords(x1, y1, (frameR, frameR), (wCam-frameR, hCam-frameR))
                        mouse.move(mx, my)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        
                    # Click: index and middle
                    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                        length = np.hypot(lmList[8][1] - lmList[12][1], lmList[8][2] - lmList[12][2])
                        if length < 40:
                            cv2.circle(img, (lmList[8][1], lmList[8][2]), 15, (0, 255, 0), cv2.FILLED)
                            mouse.click()
                
                # 3. CNN Check on ROI
                if len(bbox) == 4:
                    xmin, ymin, xmax, ymax = bbox
                    pad = 20
                    h, w, c = img.shape
                    ry1, ry2 = max(0, ymin-pad), min(h, ymax+pad)
                    rx1, rx2 = max(0, xmin-pad), min(w, xmax+pad)
                    
                    roi = img[ry1:ry2, rx1:rx2]
                    gesture = classifier.infer_cnn(roi)
                    cv2.putText(img, f'CNN: {gesture}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    
                    if gesture == 'Fist':
                        mouse.start_drag()
                    elif gesture == 'Open':
                        mouse.stop_drag()
                    elif gesture == 'ThumbUp':
                        mouse.scroll(50)
                        
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            
            cv2.imshow("AI Virtual Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
