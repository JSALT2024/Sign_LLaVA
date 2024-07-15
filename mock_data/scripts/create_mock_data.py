import h5py
import json
import numpy as np
import os
import string
import random
import shutil


def generate_random_text(length):
    vocabulary = "The job's resource requirements are specified, indicating the \
    necessary resources for the job to execute on the compute nodes. \
    These specifications include the job name, output filename, RAM capacity, \
    number of CPUs, nodes, tasks, time constraints, and other relevant parameters. \
    These commands, known as SBATCH directives, must be written in uppercase format \
    and preceded by a pound sign.".split()
    random_text = " ".join(random.sample(vocabulary, random.randint(5, 20)))
    return random_text

if __name__ == '__main__':
    output_path = '.'
    mae = h5py.File(os.path.join(output_path, 'mae.train.h5'), 'w')
    dino = h5py.File(os.path.join(output_path, 'dino.train.h5'), 'w')
    sign2vec = h5py.File(os.path.join(output_path, 'sign2vec.train.h5'), 'w')
    pose = h5py.File(os.path.join(output_path, 'pose.train.h5'), 'w')

    json_dict = {}

    video_names = ['video1', 'video2', 'video3', 'video4', 'video5']
    for video in video_names:
        num_of_clips = np.random.randint(60, 150)
        clip_names = [f'{i}' for i in range(0, num_of_clips)]
        mae_h5 = mae.create_group(video)
        dino_h5 = dino.create_group(video)
        sign2vec_h5 = sign2vec.create_group(video)
        pose_h5 = pose.create_group(video)
        json_dict[video] = {}
        for clip in clip_names:
            text_length = np.random.randint(5, 20)
            json_dict[video][clip] = {"translation": generate_random_text(text_length),
                                      "paraphrases": [generate_random_text(text_length) for _ in range(3)]}

            num_of_frames = np.random.randint(5, 30)

            features_mae = np.random.rand(num_of_frames, 768)
            mae_h5.create_dataset(clip, shape=(features_mae.shape[0], features_mae.shape[1]), dtype=np.float16)
            mae_h5[clip][:] = features_mae

            features_dino = np.random.rand(num_of_frames, 384)
            dino_h5.create_dataset(clip, shape=(features_dino.shape[0], features_dino.shape[1]), dtype=np.float16)
            dino_h5[clip][:] = features_dino

            features_sign2vec = np.random.rand(num_of_frames, 768)
            sign2vec_h5.create_dataset(clip, shape=(features_sign2vec.shape[0], features_sign2vec.shape[1]),
                                       dtype=np.float16)
            sign2vec_h5[clip][:] = features_sign2vec

            features_pose = np.random.rand(num_of_frames, 32)
            pose_h5.create_dataset(clip, shape=(features_pose.shape[0], features_pose.shape[1]), dtype=np.float16)
            pose_h5[clip][:] = features_pose

    mae.close()
    dino.close()
    sign2vec.close()
    pose.close()

    for input_type in ['mae', 'dino', 'sign2vec', 'pose']:
        shutil.copy(os.path.join(output_path, input_type+'.train.h5'), os.path.join(output_path, input_type+'.dev.h5'))

    with open(os.path.join(output_path, 'annotation.train.json'), 'w') as f:
        json.dump(json_dict, f)
    
    shutil.copy(os.path.join(output_path, "annotation.train.json"), os.path.join(output_path, "annotation.dev.json"))