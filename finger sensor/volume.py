import cv2
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities

#audio
volume = AudioUtilities.GetSpeakers().EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()

prevVol = minVol  # smoothing

# hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# loop 
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            x1, y1 = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
            x2, y2 = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)

            # real distance
            distance = math.hypot(x2 - x1, y2 - y1)

            # range 
            dist_min = 25    # fingers closed
            dist_max = 220   # fingers wide open

            vol = np.interp(distance, [dist_min, dist_max], [minVol, maxVol])
            vol = max(min(vol, maxVol), minVol)

            # smoothing
            smoothVol = prevVol + (vol - prevVol) * 0.15
            volume.SetMasterVolumeLevel(smoothVol, None)
            prevVol = smoothVol

            # UI
            vol_percent = int(np.interp(smoothVol, [minVol, maxVol], [0, 100]))

            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Distance: {int(distance)}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Volume: {vol_percent}%",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Finger Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


