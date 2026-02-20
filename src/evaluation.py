import time
import pandas as pd
import cv2
import os
from capture import VideoCaptureAsync
from tracker import HandTracker
from classifier import HybridClassifier

def benchmark_latency(num_frames=100):
    """
    Measures the latency of the three different pipelines:
    1. MediaPipe Only (Heuristics)
    2. CNN Only
    3. Hybrid (MediaPipe + CNN)
    """
    print("Initializing benchmark...")
    tracker = HandTracker(max_hands=1)
    classifier = HybridClassifier(model_path='gesture_cnn.pth')
    cap = VideoCaptureAsync(src=0).start()
    
    results = []
    
    print(f"Running benchmark for {num_frames} frames...")
    time.sleep(2) # Warm up camera
    
    for i in range(num_frames):
        grabbed, img = cap.read()
        if not grabbed or img is None:
            continue
            
        h, w, c = img.shape
        
        # 1. Pipeline: MediaPipe Only (Tracking + Heuristics)
        t0 = time.time()
        img_mp = tracker.find_hands(img.copy(), draw=False)
        lmList, _ = tracker.get_position_and_bbox(img_mp, draw=False)
        if len(lmList) > 0:
            fingers = classifier.heuristic.get_fingers_up(lmList)
        t_mp = (time.time() - t0) * 1000 
        
        # 2. Pipeline: CNN Only (Using arbitrary ROI to simulate overhead)
        t0 = time.time()
        roi_mock = img[h//4:h*3//4, w//4:w*3//4] 
        gesture = classifier.infer_cnn(roi_mock)
        t_cnn = (time.time() - t0) * 1000 
        
        # 3. Pipeline: Hybrid (MediaPipe tracking + CNN on actual ROI)
        t0 = time.time()
        img_hybrid = tracker.find_hands(img.copy(), draw=False)
        lmList_h, bbox_h = tracker.get_position_and_bbox(img_hybrid, draw=False)
        if len(bbox_h) == 4:
            xmin, ymin, xmax, ymax = bbox_h
            roi_h = img_hybrid[max(0, ymin-20):min(h, ymax+20), max(0, xmin-20):min(w, xmax+20)]
            gesture_h = classifier.infer_cnn(roi_h)
        t_hybrid = (time.time() - t0) * 1000 
        
        results.append({
            'Frame': i,
            'MediaPipe_ms': t_mp,
            'CNN_Only_ms': t_cnn,
            'Hybrid_ms': t_hybrid
        })
        
    cap.stop()
    cv2.destroyAllWindows()
    
    df = pd.DataFrame(results)
    
    print("\n--- Benchmark Results (Average Latency) ---")
    print(f"MediaPipe Only: {df['MediaPipe_ms'].mean():.2f} ms (Est. FPS: {1000/max(0.1, df['MediaPipe_ms'].mean()):.1f})")
    print(f"CNN Only: {df['CNN_Only_ms'].mean():.2f} ms (Est. FPS: {1000/max(0.1, df['CNN_Only_ms'].mean()):.1f})")
    print(f"Hybrid: {df['Hybrid_ms'].mean():.2f} ms (Est. FPS: {1000/max(0.1, df['Hybrid_ms'].mean()):.1f})")
    
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/latency_benchmark.csv', index=False)
    print("Results saved to results/latency_benchmark.csv")
    
if __name__ == '__main__':
    benchmark_latency(num_frames=50)
