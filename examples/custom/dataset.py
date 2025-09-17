# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, MessagesPreprocessor, load_dataset, register_dataset
import os


class CustomPreprocessor_Reason_test(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = super().preprocess(row)
        
        images = row['images']
        if not isinstance(images, list):
            row["images"] = [images]

        for image in row["images"]:
            if not os.path.exists(image):
                return

        import ipdb
        ipdb.set_trace()
        return row

register_dataset(
    DatasetMeta(
        # ms_dataset_id='AI-ModelScope/ShareGPT-4o',
        # hf_dataset_id='OpenGVLab/ShareGPT-4o',
        dataset_path="/test/cd/gpt4o.json",
        dataset_name="reason",
        preprocess_func=CustomPreprocessor_Reason_test(),
        # subsets=['image_caption'],
        # split=['images'],
        # tags=['vqa', 'multi-modal'],
    ))


if __name__ == '__main__':
    dataset = load_dataset(['reason'])[0]
    import ipdb
    ipdb.set_trace()
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')
