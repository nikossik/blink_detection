import numpy as np
import cv2
import dlib
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye: np.array) -> float:
    """
    This function determines how open or closed the eye is on the frame

    :param eye: eye coordinates
    :return: degree of eye openness
    """
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])

    ear = (a + b) / (2.0 * c)

    return ear


def calculate_blinks(video_path: str, detector: dlib.fhog_object_detector, predictor: dlib.shape_predictor, th_close: float=0.21, th_open: float=0.26) -> tuple:
    """
    This function counts left blinks and right blinks on the video

    :param video_path: path to video
    :param detector: detector nn
    :param predictor: predictor nn
    :return: left blinks and right blinks count
    """
    # params
    right_eye_points = list(range(36, 42))
    left_eye_points = list(range(42, 48))
    total_left = 0
    total_right = 0

    video_capture = cv2.VideoCapture(video_path)  # open video

    left_close = False
    right_close = False
    while video_capture.isOpened():  # read video by frames
        ret, frame = video_capture.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                landmarks = np.array([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[right_eye_points]  # left eye cords
                right_eye = landmarks[left_eye_points]  # right eye cords

                ear_left = eye_aspect_ratio(left_eye)   # left eye degree
                ear_right = eye_aspect_ratio(right_eye)  # right eye degree

                # count blinks
                if ear_left < th_close and not left_close:
                    left_close = True

                elif ear_left >= th_open and left_close:
                    total_left += 1
                    left_close = False

                if ear_right < th_close and not right_close:
                    right_close = True

                elif ear_right >= th_open and right_close:
                    total_right += 1
                    right_close = False
        else:
            break

    video_capture.release()  # close video

    return total_left, total_right
