"""
Frame-wise processing.
creates an Excel for each video; includes frame-wise information (one row per frame)
20 FEATURES: for all landmarks - x, y, z coordinates and Euclidean speed ;
only for the two wrists - distance from body plane.
"""

import cv2
import glob
import math
import os
import openpyxl
import pandas as pd
from scipy.spatial.distance import euclidean
import mediapipe as mp


def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """Find equation of plane from three 3D points."""
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)

    return a, b, c, d


def shortest_distance(x1, y1, z1, a, b, c, d):
    """Find distance from the plane."""
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))

    if e == 0:
        return 0

    return d / e


def get_landmarks(video_path):
    """Extracts landmarks from a video using MediaPipe Pose model."""
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    frame_num = 0
    df = pd.DataFrame(columns=['Video Name', 'Video Number', 'Person Number', 'Category', 'Label', 'Frame Number',

                               'Mid Eye X', 'Mid Eye Y', 'Mid Eye Z', 'Mid Eye Speed',

                               'Mid Clav X', 'Mid Clav Y', 'Mid Clav Z', 'Mid Clav Speed',

                               'Right Wrist X', 'Right Wrist Y', 'Right Wrist Z', 'Right Wrist Speed', 'Right Wrist Dist',

                               'Left Wrist X', 'Left Wrist Y', 'Left Wrist Z', 'Left Wrist Speed', 'Left Wrist Dist'])

    prev_frame_timestamp = None

    prev_mid_eye = [0, 0, 0]
    prev_mid_clav = [0, 0, 0]
    prev_right_wrist = [0, 0, 0]
    prev_left_wrist = [0, 0, 0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        current_frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if prev_frame_timestamp is not None:
            dt = (current_frame_timestamp - prev_frame_timestamp) / 1000.0  # Convert to seconds.
        else:
            dt = 0

        results = pose.process(frame)

        if results.pose_landmarks is None:
            mid_eye = [0, 0, 0]
            mid_clav = [0, 0, 0]
            right_wrist = [0, 0, 0]
            left_wrist = [0, 0, 0]
            right_hip = [0, 0, 0]
            left_hip = [0, 0, 0]

        else:
            left_eye = results.pose_landmarks.landmark[2]
            right_eye = results.pose_landmarks.landmark[5]
            mid_eye = [(left_eye.x + right_eye.x) / 2,
                       (left_eye.y + right_eye.y) / 2,
                       (left_eye.z + right_eye.z) / 2]

            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]
            mid_clav = [(left_shoulder.x + right_shoulder.x) / 2,
                        (left_shoulder.y + right_shoulder.y) / 2,
                        (left_shoulder.z + right_shoulder.z) / 2]

            right_wrist = [
                results.pose_landmarks.landmark[15].x,
                results.pose_landmarks.landmark[15].y,
                results.pose_landmarks.landmark[15].z
            ]

            left_wrist = [
                results.pose_landmarks.landmark[16].x,
                results.pose_landmarks.landmark[16].y,
                results.pose_landmarks.landmark[16].z
            ]

            right_hip = [
                results.pose_landmarks.landmark[24].x,
                results.pose_landmarks.landmark[24].y,
                results.pose_landmarks.landmark[24].z
            ]

            left_hip = [
                results.pose_landmarks.landmark[23].x,
                results.pose_landmarks.landmark[23].y,
                results.pose_landmarks.landmark[23].z
            ]

        # Calculate speed.
        if prev_frame_timestamp is not None and int(prev_frame_timestamp) != 0:
            mid_eye_speed = euclidean(prev_mid_eye, mid_eye) / dt
            mid_clav_speed = euclidean(prev_mid_clav, mid_eye) / dt
            right_wrist_speed = euclidean(prev_right_wrist, right_wrist) / dt
            left_wrist_speed = euclidean(prev_left_wrist, left_wrist) / dt

        else:
            mid_eye_speed = 0
            mid_clav_speed = 0
            right_wrist_speed = 0
            left_wrist_speed = 0

        prev_frame_timestamp = current_frame_timestamp

        # The body plane.
        a, b, c, d = equation_plane(x1=mid_clav[0], y1=mid_clav[1], z1=mid_clav[2],
                                    x2=right_hip[0], y2=right_hip[1], z2=right_hip[2],
                                    x3=left_hip[0], y3=left_hip[1], z3=left_hip[2])

        right_wrist_dist = shortest_distance(right_wrist[0], right_wrist[1], right_wrist[2],
                                             a, b, c, d)
        left_wrist_dist = shortest_distance(left_wrist[0], left_wrist[1], left_wrist[2],
                                            a, b, c, d)

        #### Append data to df ####
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        file = video_name
        file_info = file.split("_")
        video_num = file_info[1].lstrip("0")
        emotion = file_info[2].rsplit(".", 1)[0]  # Split from the right end and remove file extension.

        if "hearing" in file:
            category = "hearing"
            person_num = file_info[0][3:]
        elif "D" in file:
            category = "deaf"
            person_num = file_info[0][1:]
        elif "C" in file and "sign" in file:
            category = "CODA sign"
            person_num = file_info[0][1:]
        elif "C" in file and "speech" in file:
            category = "CODA speech"
            person_num = file_info[0][1:]
        else:
            category = "hearing"
            person_num = file_info[0][3:]
        person_num = person_num.lstrip("0")

        df = df._append({'Video Name': video_name,
                        'Video Number': video_num,
                        'Person Number': person_num,
                        'Category': category,
                        'Label': emotion,
                        'Frame Number': frame_num,

                        'Mid Eye X': mid_eye[0],
                        'Mid Eye Y': mid_eye[1],
                        'Mid Eye Z': mid_eye[2],
                        'Mid Eye Speed': mid_eye_speed,

                        'Mid Clav X': mid_clav[0],
                        'Mid Clav Y': mid_clav[1],
                        'Mid Clav Z': mid_clav[2],
                        'Mid Clav Speed': mid_clav_speed,

                        'Right Wrist X': right_wrist[0],
                        'Right Wrist Y': right_wrist[1],
                        'Right Wrist Z': right_wrist[2],
                        'Right Wrist Speed': right_wrist_speed,
                        'Right Wrist Dist': right_wrist_dist,

                        'Left Wrist X': left_wrist[0],
                        'Left Wrist Y': left_wrist[1],
                        'Left Wrist Z': left_wrist[2],
                        'Left Wrist Speed': left_wrist_speed,
                        'Left Wrist Dist': left_wrist_dist
                        },
                       ignore_index=True)

        # Update frame timestamps.
        prev_frame_timestamp = current_frame_timestamp
        prev_mid_eye = mid_eye
        prev_mid_clav = mid_clav
        prev_right_wrist = right_wrist
        prev_left_wrist = left_wrist
    return df


video_folder = r'C:\Users\USER\Downloads\DataForHana'
output_folder = r'C:\Users\USER\Downloads\HBC_all_vids\RowPerFrame'

video_files = glob.glob(os.path.join(video_folder, '*.mp4'))

for video_file in video_files:
    df = get_landmarks(video_file)
    video_name_ = os.path.splitext(os.path.basename(video_file))[0]
    output_file = os.path.join(output_folder, f'{video_name_}.xlsx')
    df.to_excel(output_file, index=False)

    print(f'Saved landmarks for video {video_file} to {output_file}')