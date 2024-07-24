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
VIDEO_PLACEHOLDER = "<video>"
# Baseline: "<video_start>vf1<video_end><video_start>vf2<video_end>"
# "the first visual feature stream: "<video_start>vf1<video_end>"the second visual feature stream: "<video_start>vf2<video_end>
INPUT_TYPES = ["mae", "sign2vec", "dino", "pose"]
VIDEO_TOKEN_INDEX = -200
#PROMPT_CONTEXT = "Here are some preceding sentences as context: <context>. Translate the next sentence given by the American Sign Language video into English. "
#PROMPT_CONTEXT = "Given some of the preceding sentences as context, translate the given American Sign Language video into English. "
#PROMPT_NO_CONTEXT = "Translate the given American Sign Language video into English. "

PROMPT_OPTIONS = {
    "translate_with_context": "Here are some preceding sentences as context: '<context>'. Translate the next sentence given by the American Sign Language video into English. ",
    "translate_no_context": "Translate the given American Sign Language video into English. ",
    "one_word_present": "Given the ASL video input, answer the following question with 'yes' or 'no': Is the sign '<word>' present? ",
    "multi_words_present": "Given the ASL video input, answer the following questions with 'yes' or 'no' for each sign listed, separated by commas: Are the signs '<words>' present?",
    "is_reversed": "Given the video input, answer the following question with 'yes' or 'no': Is the video presented in reversed temporal order?"
}