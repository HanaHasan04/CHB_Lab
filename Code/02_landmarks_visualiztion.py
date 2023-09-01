"""
Beautiful landmarks' visualization. (NO faces or bodies included)
3 GIF options:
- The first GIF shows the complete set of captured landmarks from MediaPipe Pose,
  along with the underlying skeleton.
- The second GIF focuses solely on our four points of interest, offering a closer examination.
- The third GIF presents the four landmarks in all frames of the video,
  providing a comprehensive view of their movements and patterns throughout the entire sequence.
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# videos dir
input_video_path = r"C:\Users\USER\Downloads\HBC_Data4Hana_all\C01_14_angry_speech.mp4"

# output GIF path
output_gif_path = r"C:\Users\USER\Documents\UniversityProjects\Hagit_Lab\webpage\webpage_files\mediapipe_examples\deaf\our_points___.gif"

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []

white_background = np.ones((height, width, 3), dtype=np.uint8) * 255    # white background

# mediapipe
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
draw_landmarks = mp.solutions.drawing_utils.draw_landmarks

# store the landmarks of previous frames
previous_landmarks = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(frame_rgb)
    background = white_background.copy()

    if results.pose_landmarks is not None:

        # define our landmarks of interest
        landmarks = results.pose_landmarks.landmark
        right_wrist = (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * width),
                       int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y * height))
        left_wrist = (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x * width),
                      int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y * height))
        right_shoulder = (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                          int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * height))
        left_shoulder = (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * width),
                         int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * height))
        right_eye = ((int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].x * width) +
                      int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER].x * width)) // 2,
                     (int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].y * height) +
                      int(landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER].y * height)) // 2)
        left_eye = ((int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].x * width) +
                     int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER].x * width)) // 2,
                    (int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].y * height) +
                     int(landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER].y * height)) // 2)

        mid_shoulders = ((int((landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x +
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x) / 2 * width)),
                         (int((landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y +
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y) / 2 * height)))

        mid_eyes = ((int((landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].x +
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].x) / 2 * width)),
                         (int((landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].y +
                               landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].y) / 2 * height)))

        # GIF Option 1: all landmarks + skeleton
        ####################################################################################
        if results.pose_landmarks is not None:
            draw_landmarks(
                background, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), circle_radius=5),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=2)
            )
        ####################################################################################

        # GIF Option 2: four points of interest, frame-by-frame
        ####################################################################################
        cv2.circle(background, right_wrist, radius=8, color=(0, 0, 255), thickness=-1)
        cv2.circle(background, left_wrist, radius=8, color=(0, 0, 255), thickness=-1)
        cv2.circle(background, mid_shoulders, radius=8, color=(0, 0, 255), thickness=-1)
        cv2.circle(background, mid_eyes, radius=8, color=(0, 0, 255), thickness=-1)
        ####################################################################################

        # GIF Option 3: four points of interest, in all frames of the video (show all prev. frames)
        ####################################################################################
        # append the landmarks to the list
        previous_landmarks.append(landmarks)
        # draw landmarks from previous frames
        for prev_landmarks in previous_landmarks:
            cv2.circle(background, (int(prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * width),
                                    int(prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y * height)),
                       radius=8, color=(255, 0, 255), thickness=-1)
            cv2.circle(background, (int(prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x * width),
                                    int(prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y * height)),
                       radius=8, color=(0, 255, 255), thickness=-1)
            cv2.circle(background, (int((prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x +
                                          prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x) / 2 * width),
                                    int((prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y +
                                          prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y) / 2 * height)),
                       radius=8, color=(0, 0, 255), thickness=-1)
            cv2.circle(background, (int((prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].x +
                                          prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].x) / 2 * width),
                                    int((prev_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER].y +
                                          prev_landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER].y) / 2 * height)),
                       radius=8, color=(0, 255, 0), thickness=-1)
        ####################################################################################

    frames.append(Image.fromarray(background))

    cv2.imshow("Output", background)
    if cv2.waitKey(1) == ord('q'):
        break

frames[0].save(output_gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=int(1000 / fps), loop=0)
cap.release()
cv2.destroyAllWindows()
