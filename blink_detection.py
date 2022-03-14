import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def calculate_blinks(video_path):
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))

    th_close = 0.21
    th_open = 0.26

    TOTAL_LEFT = 0
    TOTAL_RIGHT = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    video_capture = cv2.VideoCapt ure(video_path)

    left_close = False
    right_close = False
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[LEFT_EYE_POINTS]
                right_eye = landmarks[RIGHT_EYE_POINTS]

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                if ear_left < th_close and not left_close:
                    left_close = True

                elif ear_left >= th_open and left_close:
                    TOTAL_LEFT += 1
                    left_close = False

                if ear_right < th_close and not right_close:
                    right_close = True

                elif ear_right >= th_open and right_close:
                    TOTAL_RIGHT += 1
                    right_close = False
        else:
            break

    video_capture.release()

    return TOTAL_LEFT, TOTAL_RIGHT
