from llava.sign_public_api import SignLlava, SignLlavaInput, \
    SignLlavaOutput, GenerationConfig, prepare_translation_prompt
import numpy as np


def test_sign_llava():
    print("Loading the Sign LLaVA model...")
    our_model = SignLlava.load_from_checkpoint("/export/fs06/xzhan138/Sign_LLaVA/signllava/checkpoints/test_ckpt_July_26_2024_11am")
    print("Model loaded successfully!")
    input_data = SignLlavaInput(
        sign2vec_features=np.zeros(shape=(150, 768), dtype=np.float32),
        mae_features=np.zeros(shape=(300, 768), dtype=np.float32),
        dino_features=np.zeros(shape=(300, 1152), dtype=np.float32),
        prompt=prepare_translation_prompt(context=None),
        generation_config=GenerationConfig()
    )
    output_data: SignLlavaOutput = our_model.run_inference(input_data)

    print(output_data.output)
    #print(output_data.text_embeddings)
    print(output_data.mae_embeddings)


if __name__ == "__main__":
    test_sign_llava()