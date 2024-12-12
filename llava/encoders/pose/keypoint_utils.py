import numpy as np
from .normalization import local_keypoint_normalization, global_keypoint_normalization


def _get_keypoints(json_data: dict, data_key: str = 'cropped_keypoints'):
    right_hand_landmarks = []
    left_hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []

    keypoints = json_data[data_key]
    for frame_id in range(len(keypoints)):
        if len(keypoints[frame_id]['pose_landmarks']) == 0:
            pose_landmarks.append(np.zeros((33, 2)))
        else:
            pose_landmarks.append(np.array(keypoints[frame_id]['pose_landmarks']))

        if len(keypoints[frame_id]['right_hand_landmarks']) == 0:
            right_hand_landmarks.append(np.zeros((21, 2)))
        else:
            right_hand_landmarks.append(np.array(keypoints[frame_id]['right_hand_landmarks']))

        if len(keypoints[frame_id]['left_hand_landmarks']) == 0:
            left_hand_landmarks.append(np.zeros((21, 2)))
        else:
            left_hand_landmarks.append(np.array(keypoints[frame_id]['left_hand_landmarks']))

        if len(keypoints[frame_id]['face_landmarks']) == 0:
            face_landmarks.append(np.zeros((478, 2)))
        else:
            face_landmarks.append(np.array(keypoints[frame_id]['face_landmarks']))

    pose_landmarks = np.array(pose_landmarks)[:, :25]
    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def get_keypoints(
    keypoints: dict,
    data_key: str,
    face_landmarks_idx: list,
    normalization_methods: list
) -> np.array:
    keypoints = _get_keypoints(keypoints, data_key=data_key)
    pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = keypoints
    joints = {
        'face_landmarks': np.array(face_landmarks)[:, face_landmarks_idx, :],
        'left_hand_landmarks': np.array(left_hand_landmarks),
        'right_hand_landmarks': np.array(right_hand_landmarks),
        'pose_landmarks': np.array(pose_landmarks)
    }

    if normalization_methods:
        local_landmarks = {}
        global_landmarks = {}

        for idx, landmarks in enumerate(normalization_methods):
            prefix, landmarks = landmarks.split("-")
            if prefix == "local":
                local_landmarks[idx] = landmarks
            elif prefix == "global":
                global_landmarks[idx] = landmarks

        # local normalization
        for idx, landmarks in local_landmarks.items():
            normalized_keypoints = local_keypoint_normalization(joints, landmarks, padding=0.2)
            local_landmarks[idx] = normalized_keypoints

        # global normalization
        additional_landmarks = list(global_landmarks.values())
        if "pose_landmarks" in additional_landmarks:
            additional_landmarks.remove("pose_landmarks")

        keypoints, additional_keypoints = global_keypoint_normalization(
            joints,
            "pose_landmarks",
            additional_landmarks
        )

        for k, landmark in global_landmarks.items():
            if landmark == "pose_landmarks":
                global_landmarks[k] = keypoints
            else:
                global_landmarks[k] = additional_keypoints[landmark]

        all_landmarks = {**local_landmarks, **global_landmarks}
        data = []
        for idx in range(len(normalization_methods)):
            data.append(all_landmarks[idx])

        data = np.concatenate(data, axis=1)
    else:
        data = [joints["pose_landmarks"], joints["right_hand_landmarks"], joints["left_hand_landmarks"],
                joints["face_landmarks"]]
        data = np.concatenate(data, axis=1)
    data = data.reshape(data.shape[0], -1)

    return data
