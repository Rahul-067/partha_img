import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================
# 🔌 ARDUINO SETUP
# ============================================
import serial

ARDUINO_PORT = "COM4"   # <<< CHANGE THIS
ARDUINO_BAUD = 9600

try:
    arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD)
    time.sleep(2)
    print(f"[OK] Arduino connected on {ARDUINO_PORT}")
except:
    print("[WARNING] Arduino not detected. Running without hardware.")
    arduino = None

LAST_SIGNAL = -1  # Prevent spam

def send_to_arduino(value):
    global LAST_SIGNAL
    if arduino is None:
        return
    if LAST_SIGNAL != value:
        arduino.write(str(value).encode())
        LAST_SIGNAL = value
        print(f"[ARDUINO] Sent: {value}")


# ============================================
# YOLO Argument Parser
# ============================================
parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record


# ============================================
# Load YOLO Model
# ============================================
if not os.path.exists(model_path):
    print("ERROR: Model not found!")
    sys.exit(0)

model = YOLO(model_path)
labels = model.names


# ============================================
# INPUT SOURCE PARSING
# ============================================
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.mp4','.avi','.mkv','.mov','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'

elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print("Unsupported file format!")
        sys.exit(0)

elif "usb" in img_source:
    source_type = "usb"
    usb_idx = int(img_source.replace("usb",""))

else:
    print("Invalid source!")
    sys.exit(0)


# ============================================
# DISPLAY RESOLUTION PARSING
# ============================================
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split("x"))


# ============================================
# RECORDING SETUP
# ============================================
if record:
    if source_type not in ["video","usb"]:
        print("Recording only valid for video or USB camera.")
        sys.exit(0)

    if not user_res:
        print("Specify --resolution for recording.")
        sys.exit(0)

    recorder = cv2.VideoWriter(
        "demo1.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )


# ============================================
# INITIALIZE CAMERA OR FILE INPUT
# ============================================
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + "/*") if os.path.splitext(f)[1].lower() in img_ext_list]

elif source_type in ['video','usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)


# ============================================
# INFERENCE LOOP
# ============================================
TARGET_CLASS = "Partha"   # <<< CHANGE THIS TO YOUR CLASS

img_count = 0
frame_rate_buffer = []
fps_avg_len = 100
avg_frame_rate = 0

while True:
    t_start = time.perf_counter()

    # Load frame depending on source
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    else:
        ret, frame = cap.read()
        if not ret: break

    # Resize if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    detected_classnames = []

    # Draw detections
    for det in detections:
        classid = int(det.cls)
        classname = labels[classid]
        conf = float(det.conf)

        if conf < min_thresh: 
            continue

        # Extract bounding box
        xyxy = det.xyxy.cpu().numpy().astype(int).flatten()
        xmin, ymin, xmax, ymax = xyxy

        # Draw box
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
        cv2.putText(frame, f"{classname} {conf*100:.1f}%", 
                    (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        
        detected_classnames.append(classname)


    # ============================================
    # 🔥 ARDUINO CONTROL BASED ON YOLO DETECTION
    # ============================================
    if TARGET_CLASS in detected_classnames:
        send_to_arduino(1)   # Face detected → Light ON
    else:
        send_to_arduino(0)   # Not detected → Light OFF


    # ============================================
    # DISPLAY RESULTS
    # ============================================
    cv2.putText(frame, f"Objects: {len(detections)}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)

    cv2.imshow("YOLO detection results", frame)

    if record: recorder.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'): break


    # FPS calculation
    t_stop = time.perf_counter()
    fps = 1/(t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)
    avg_frame_rate = np.mean(frame_rate_buffer)


# ============================================
# CLEANUP
# ============================================
if source_type in ['video','usb']:
    cap.release()

if record: recorder.release()

cv2.destroyAllWindows()

print(f"Average FPS: {avg_frame_rate:.2f}")
