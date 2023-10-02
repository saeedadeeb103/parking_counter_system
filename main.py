import cv2
from utils.utils import Utils, Utils2
import numpy as np

def calc_diff(img1 , img2):
    return np.abs(np.mean(img1) - np.mean(img2))

video_path = 'resources/parking_1920_1080_loop.mp4'
mask = 'resources/mask_1920_1080.png'
mask = cv2.imread(mask, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = connected_components
slots = []
frame_nmr = 0
coef = 1
for i in range(1, totalLabels):
    x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
    y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
    h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
    w = int(values[i, cv2.CC_STAT_WIDTH] * coef)

    slots.append([x1, y1, w, h])

spots_status = [None for j in slots]
steps = 30  # Process every 3rd frame
cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#utils = Utils(conf_threshold=0.4)  # Adjust the confidence threshold as needed
utils = Utils2()
diff = [None for j in slots]
previous_frame = None
empty = 0

while True:
    success, frame = cap.read()

    if frame_nmr % steps == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(slots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]

            diff[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1+h, x1:x1+w, :])

    if frame_nmr % steps == 0:
        if previous_frame is None:
            arr_ = range(len(slots))
        else:
            arr_ = [j for j in np.argsort(diff) if diff[j] / np.amax(diff) > 0.4]
        for spot_index in arr_:
            spot = slots[spot_index]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            spot_status = utils.empty_or_not(spot_bgr=spot_crop)
            spots_status[spot_index] = spot_status
    
    if frame_nmr % steps == 0:
        previous_frame = frame.copy()
        
    
    for spot_index, spot in enumerate(slots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = slots[spot_index]
        if spot_status == "Empty":
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 255, 255), 1)
        elif spot_status == "Not Empty":
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 1)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 1)
            
    cv2.putText(frame, 'Available spots: {} / {}'.format(len([i for i in spots_status if i == "Empty"]), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('Video Frames', cv2.WINDOW_NORMAL)
    cv2.imshow("Video Frames", frame)

    if cv2.waitKey(1000 // frame_rate) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
