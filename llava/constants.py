CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

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
INPUT_TYPES = ["mae", "sign2vec", "dino", "keypoint"]
VIDEO_TOKEN_INDEX = -200
PROMPT = "Given the preceding sentences as context, translate the American sign language video into English. "