CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Name of the files used for checkpoints
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_START_TOKEN = "<video_start>"
DEFAULT_VIDEO_END_TOKEN = "<video_end>"
# Baseline: "<video_start>vf1<video_end><video_start>vf2<video_end>"
# "the first visual feature stream: "<video_start>vf1<video_end>"the second visual feature stream: "<video_start>vf2<video_end>
INPUT_TYPES = ["mae", "sign2vec", "dino", "pose"]
VIDEO_TOKEN_INDEX = -200
PROMPT = "Given some of the preceding sentences as context, translate the American Sign Language video into English. "
#PROMPT_CONTEXT = "Given some of the preceding sentences as context, translate the American Sign Language video into English."
#PROMPT_NO_CONTEXT = "Translate the American Sign Language video into English. "