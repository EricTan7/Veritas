import os
from typing import Any, Dict, List, Union

import json

from swift.llm import InferRequest
import base64


class PRM:

    def __call__(self, **kwargs) -> List[Any]:
        raise NotImplementedError


SYSTEM = """
You are a process reward model, give the reward value of the answer, you must follow the instructions below:

1. Output a float reward value between -1.0 and 1.0, -1.0 means the worst answer, 1.0 means the best answer, please think step by step to give your reasons and thoughts, but the reward must appare at the end with this format: **Reward: your-reward-value**.

2. The answer may be incomplete, you must give the reward by the existing part of the answer, taking into account semantic coherence, logical correctness, and clarity.

3. A ground truth answer will be given to you, it may be not the best one, consider it as a reference example.

Begin!
""" # noqa

QUERY = """
The original question or the previous conversation:

#query#

Here is the ground truth as the reference:

#ground_truth#

Given the upper information, give your reward(-1.0~1.0) of the following answer:

#response#
"""


class QwenMaxPRM(PRM):

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        # TODO: check request_config
        rewards = []

        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        for request, ground_truth in zip(infer_requests, ground_truths):
            previous = request['messages'][:-1]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert request['messages'][-1]['role'] == 'assistant'
            query = QUERY.replace('#query#', json.dumps(previous))
            query = query.replace('#ground_truth#', ground_truth)
            query = query.replace('#response#', request['messages'][-1]['content'])
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM
                },
                {
                    'role': 'user',
                    'content': query
                },
            ]
            completion = client.chat.completions.create(
                model='qwen-max',
                messages=messages,
            )

            content = completion.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(0.)
            else:
                try:
                    reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                    rewards.append(reward)
                except Exception:
                    rewards.append(0.)

        return rewards


class ClientPRM(PRM):

    def __init__(self, api_key=None, base_url=None, model=None):
        from swift.llm import InferClient
        import os
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        if base_url is None:
            base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        if model is None:
            model = 'qwen-plus'
        self.infer_engine = InferClient(base_url=base_url, api_key=api_key)
        self.infer_engine.strict = False
        self.infer_kwargs = {
            'model': model,
        }

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        prm_infer_requests = []
        request_config = kwargs.get('request_config')
        for request, ground_truth in zip(infer_requests, ground_truths):
            previous = request['messages'][:-1]
            if previous[0]['role'] == 'system':
                previous = previous[1:]

            assert request['messages'][-1]['role'] == 'assistant'
            query = QUERY.replace('#query#', json.dumps(previous))
            query = query.replace('#ground_truth#', ground_truth)
            query = query.replace('#response#', request['messages'][-1]['content'])
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM
                },
                {
                    'role': 'user',
                    'content': query
                },
            ]

            prm_infer_requests.append(InferRequest(messages=messages))

        responses = self.infer_engine.infer(prm_infer_requests, request_config=request_config, **self.infer_kwargs)
        rewards = []
        for response in responses:
            content = response.choices[0].message.content
            if 'Reward:' not in content:
                rewards.append(0.)
            else:
                try:
                    reward = float(content.split('Reward:')[1].strip().replace('*', ''))
                    rewards.append(reward)
                except Exception:
                    rewards.append(0.)
        return rewards





SYSTEM = "You are a helpful assistant."


class UnifiedPRM(PRM):
    def __init__(self, api_key=None, base_url=None, model=None):
        from swift.llm import InferClient
        import os
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        if base_url is None:
            base_url = "http://localhost:8003/v1"
        if model is None:
            model = "/path/to/UnifiedReward-qwen-3b"
        self.infer_engine = InferClient(base_url=base_url, api_key=api_key)
        self.infer_engine.strict = False
        self.infer_kwargs = {
            'model': model,
        }

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        prm_infer_requests = []
        request_config = kwargs.get('request_config')
        for messages, paths in zip(kwargs["messages"], kwargs["images"]):
            image_path = paths[0]["path"]

            assert messages[1]["role"] == "user"
            assert messages[-1]["role"] == "assistant"
            prompt_str = messages[1]["content"]
            reasoning_str = messages[-1]["content"]

            prompt = f"""
You are provided with an image and a question for this image. The given image is an AI-generated image, and the provided response is the reasoning process of determining its authenticity.
You should re-examine the image carefully, and then review the self-reflection content enclosed in the <reflection> </reflection> tags:
1. Is the reflection content redundant with the reasoning content? Is the reflection content just a restatement or conclusion of previous reasoning? The reflection should introduce new insights rather than restatement.
2. The reflection should not be vague statements such as "too perfect" or "lack of imperfections". Instead, it should be specific and detailed.

From 0 to 100, how much do you rate for the reflection quality?
Be strict, give low score if it is not aligned with the above principles.
Provide a few lines for explanation and the rate number at last after "Final Score:".

Your task is provided as follows:

Question: [{prompt_str}]
Response: [{reasoning_str}]
"""

            with open(image_path, "rb") as f:
                image_encoded = base64.b64encode(f.read()).decode("utf-8")
            base64_image = f"data:image;base64,{image_encoded}"
            messages = [
                {
                    'role': 'system',
                    'content': SYSTEM
                },
                {
                    'role': 'user',
                    'content': [
                        {"type": "image_url", "image_url": {"url": base64_image}},
                        {"type": "text", "text": prompt}   
                    ]
                },
            ]

            prm_infer_requests.append(InferRequest(messages=messages))

        responses = self.infer_engine.infer(prm_infer_requests, request_config=request_config, **self.infer_kwargs)
        rewards = []
        for i, response in enumerate(responses):
            try:
                content = response.choices[0].message.content
                content = content.strip()
                try:
                    content = content.split("Final Score:")[-1].strip()
                    content = float(content)
                except:
                    try:
                        content = content[:2]
                        content = float(content)
                    except:
                        # Process timeout error
                        print("PRM failed")
                        content = 0.0
                if content <= 70:
                    reward = -0.5
                elif content < 90:
                    reward = 0.0
                else:
                    reward = 0.5
                if "</reflection>" in kwargs["messages"][i][-1]["content"]:
                    rewards.append(reward)
                else:
                    rewards.append(0.0)
            except:
                rewards.append(0.0)
        return rewards



prms = {
    'qwen_max': QwenMaxPRM,
    'client': ClientPRM,
    'unifiedprm': UnifiedPRM
}