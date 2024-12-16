from typing import List, Tuple

# import cv2
from PIL import Image
import numpy as np


def crop_frame(image, bounding_box):
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bounding_box
    # fix box if out of the frame
    if x0 < 0:
        x1 += np.abs(x0)
        x0 = 0
    if y0 < 0:
        y1 += np.abs(y0)
        y0 = 0
    if x1 > w:
        dif = x1 - w
        x0 -= dif
        x1 = w
    if y1 > h:
        dif = y1 - h
        y0 -= dif
        y1 = h
    cropped_frame = image[y0:y1, x0:x1]
    return cropped_frame


def resize_frame(frame, frame_size):
    if frame is not None and frame.size > 0:
            frame = Image.fromarray(frame)
            frame = frame.resize(frame_size, resample=Image.Resampling.LANCZOS)
            return np.array(frame) #cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
    else:
        return None


def get_local_crops(images: List[np.array], input_predictions: dict) -> Tuple[list, list, list]:
    prev_face_frame = None
    prev_hand1_frame = None
    prev_hand2_frame = None

    out_face = []
    out_hand1 = []
    out_hand2 = []

    # check if video and keypoints have same number of frames
    frames = len(images)
    keypoints = len(input_predictions["bbox_face"])
    if frames != keypoints:
        print("[get_local_crops]: inconsistent number of frames and predictions")

    for i in range(np.min([frames, keypoints])):
        frame = images[i]

        # it contains a body pose that can be use as reference to get the face and hands
        if input_predictions["bbox_face"][i]:
            face_bbox = input_predictions["bbox_face"][i]
            face_frame = crop_frame(frame, face_bbox)
            face_frame = resize_frame(face_frame, (56, 56))
            out_face.append(face_frame)
            prev_face_frame = face_frame
        elif prev_face_frame is not None:
            out_face.append(prev_face_frame)
        else:
            face_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_face.append(face_frame)

        if input_predictions["bbox_left_hand"][i]:
            hand1_bbox = input_predictions["bbox_left_hand"][i]
            hand1_frame = crop_frame(frame, hand1_bbox)
            hand1_frame = resize_frame(hand1_frame, (56, 56))
            out_hand1.append(hand1_frame)
            prev_hand1_frame = hand1_frame
        elif prev_hand1_frame is not None:
            out_hand1.append(prev_hand1_frame)
        else:
            hand1_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand1.append(hand1_frame)

        if input_predictions["bbox_right_hand"][i]:
            hand2_bbox = input_predictions["bbox_right_hand"][i]
            hand2_frame = crop_frame(frame, hand2_bbox)
            hand2_frame = resize_frame(hand2_frame, (56, 56))
            out_hand2.append(hand2_frame)
            prev_hand2_frame = hand2_frame
        elif prev_hand2_frame is not None:
            out_hand2.append(prev_hand2_frame)
        else:
            hand2_frame = np.zeros((56, 56, 3), dtype=np.uint8)
            out_hand2.append(hand2_frame)

    return out_face, out_hand1, out_hand2  # face, hand_left, hand_right
