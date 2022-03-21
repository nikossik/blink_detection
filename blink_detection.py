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


def calculate_ears(video_path: str, detector: dlib.fhog_object_detector, predictor: dlib.shape_predictor) -> tuple:
    """
    This function calculates left and right ears

    :param video_path: path to video
    :param detector: detector nn
    :param predictor: predictor nn
    :return: left ears and right ears
    """

    # params
    right_eye_points = list(range(36, 42))
    left_eye_points = list(range(42, 48))

    left_ears = []
    right_ears = []

    video_capture = cv2.VideoCapture(video_path)  # open video

    while video_capture.isOpened():  # read video by frames
        ret, frame = video_capture.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                landmarks = np.array([[p.x, p.y] for p in predictor(frame, rect).parts()])

                left_eye = landmarks[right_eye_points]  # left eye cords
                right_eye = landmarks[left_eye_points]  # right eye cords

                ear_left = eye_aspect_ratio(left_eye)  # left eye degree
                ear_right = eye_aspect_ratio(right_eye)  # right eye degree

                left_ears.append(ear_left)
                right_ears.append(ear_right)

        else:
            break

    video_capture.release()  # close video

    return left_ears, right_ears


def calculate_blinks(left_ears: list, right_ears: list, th_close: float = 0.21, th_open: float = 0.26) -> tuple:
    """
    This function counts left blinks and right blinks on the video

    :param left_ears: left ears
    :param right_ears: right ears
    :param th_close: threshold for close eye
    :param th_open: threshold for open eye
    :return: left and right blinks
    """

    left_blinks = 0
    right_blinks = 0

    # calculate for left eye
    eye_close = False
    for ear in left_ears:
        if ear < th_close and not eye_close:
            eye_close = True

        elif ear >= th_open and eye_close:
            eye_close = False
            left_blinks += 1

    # calculate for right eye
    eye_close = False
    for ear in right_ears:
        if ear < th_close and not eye_close:
            eye_close = True

        elif ear >= th_open and eye_close:
            eye_close = False
            right_blinks += 1

    return left_blinks, right_blinks
