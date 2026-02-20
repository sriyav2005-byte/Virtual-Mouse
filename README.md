# Virtual-Mouse
Virtual Mouse is a real-time gesture-controlled system.It features a high-performance hybrid pipeline that leverages MediaPipe for ultra-fast hand tracking and simple heuristics, and a PyTorch CNN for complex gesture recognition. This architecture optimizes the balance between real-time inference speed and accuracy, fulfilling the core requirements of your research project.

-------------------------------------------
# activate environment
.\venv\Scripts\activate

# sample collect of fist gesture
python src/train_cnn.py collect Fist

# train model after data collection
python src/train_cnn.py train

# to run the mouse
python src/main.py

# to run research benchmark
python src/evaluation.py
