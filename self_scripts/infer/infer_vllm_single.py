import base64
from openai import OpenAI
import io
import os
import json
import argparse
from tqdm import tqdm
from cal_metrics import get_metrics



## System prompt
# zero-shot Deepfake Detection prompt
df_prompt = """You are given an facial image. Please analyze the provided facial image and determine whether it is authentic or fake based on the following classification criteria:

Real Captured Facial image
    - Images captured using a real camera or device without any alternations or manipulation.
    
Fake Facial Image
    - Images generated or manipulated using digital technologies, such as deepfakes, face swapping, face reenactment, photo editing software, entire face synthesis, etc.

Output the thinking process in <think> </think> and final answer ("real" or "fake") in <answer> </answer> tags, i.e., the output answer format should be as follows:
<think> your thinking process here </think> <answer> your judgement here </answer>
Please strictly follow the format."""

# R1 prompt
r1_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"""



## User prompt
problem = """<image> Please determine the authenticity of this image."""
problem_mimo = """<image> Please determine the authenticity of this image. Output your final answer ("real" or "fake") in <answer> </answer> tags"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VLLM Inference')
    parser.add_argument('--image_path', type=str,
                        default='',
                        help='path to your image')
    parser.add_argument('--model', type=str,
                        default='model',
                        help='model type')
    args = parser.parse_args()

    openai_api_key = "EMPTY"
    vllm_server_config = {
        ## Change your model path here
        "model": {
            "api": "http://localhost:8000/v1",
            "model": "/path/to/your/model"
        },
        "mimo-7b": {
            "api": "http://localhost:8000/v1",
            "model": "checkpoints/Mo-VL-7B-RL"
        },
        "qwen-7b": {
            "api": "http://localhost:8001/v1",
            "model": "checkpoints/Qwen2.5-VL-7B-Instruct"
        },
        "intern-8b": {
            "api": "http://localhost:8002/v1",
            "model": "checkpoints/InternVL3-8B"
        },
        "glm-9b-think": {
            "api": "http://localhost:8003/v1",
            "model": "checkpoints/GLM-4.1V-9B-Thinking"
        }
    }
    test_model = args.model
    config = vllm_server_config[test_model]

    openai_api_base = config["api"]
    model_name = config["model"].split("/")[-1]

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )


    image_path = args.image_path

    if "MiMo-VL" in model_name:
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        prompt = problem_mimo
    else:
        system_prompt = df_prompt
        prompt = problem

    with open(image_path, "rb") as f:
        image_encoded = base64.b64encode(f.read()).decode("utf-8")
    base64_fake_image = f"data:image;base64,{image_encoded}"

    try:
        chat_response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", "image_url": {"url": base64_fake_image},
                        },
                        {
                            "type": "text", "text": prompt
                        }
                    ],
                },
            ],
            temperature=0.6
        )

        full_content = chat_response.choices[0].message.content

        print(image_path)
        print(full_content.strip())
    except:
        print("failed: ", image_path)
