import os
import json
import torch
from tqdm import tqdm
torch.manual_seed(1234)
import argparse
import functools
import multiprocessing as mp
from multiprocessing import Pool

from transformers import AutoProcessor, AutoModelForCausalLM, Glm4vForConditionalGeneration
from cal_metrics import get_metrics



def run_predict(rank, world_size, val_set, args):
    if "GLM-4.1V" in args.model_path:
        model = Glm4vForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=f"cuda:{rank}"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=f"cuda:{rank}"
        )
    processor = AutoProcessor.from_pretrained(args.model_path)
    if rank == 0:
        print(f"Model Loaded from: {args.model_path}")
    model = model.eval()

    rank = rank
    world_size = world_size
    import math
    split_length = math.ceil(len(val_set)/world_size)
    split_images = val_set[int(rank*split_length) : int((rank+1)*split_length)]

    # prepare inputs for all data
    messages = []
    images, labels = [], []
    for item in split_images:
        if isinstance(item["images"], list):
            img_path = item["images"][0]
        else:
            img_path = item["images"]
        
        prompt = """<image> Please determine the authenticity of this image. Output your final answer ("real" or "fake") in <answer> </answer> tags"""
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": img_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        messages.append(message)
        images.append(img_path)
        labels.append(item["label"])


    # ******************** batch inference
    batch_size = args.batch_size
    all_outputs = []
    for i in tqdm(range(0, len(messages), batch_size)):
        try:
            batch_messages = messages[i:i + batch_size]
            batch_images = images[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
        except:
            batch_messages = messages[i:]   # last batch
            batch_images = images[i:]
            batch_labels = labels[i:]

        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for j in range(len(batch_output_text)):
            all_outputs.append({
                "images": batch_images[j],
                "result": batch_output_text[j],
                "label": batch_labels[j]
            })

    return all_outputs


def main(val_set, args, data_file):
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    with Pool(world_size) as pool:
        func = functools.partial(run_predict, world_size=world_size, val_set=val_set, args=args)
        result_lists = pool.map(func, range(world_size))

    results_all = []
    for i in range(world_size):
        results_all.extend(result_lists[i])
    
    # save
    model_name = args.model_path.split("/")[-1]
    save_path = data_file.replace(args.data_dir, f"./results/{model_name}/infer_result")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results_all, f, indent=4)

    if args.metrics:
        result = get_metrics(results_all)
        print(f"[{args.data_file}]  acc: {result['acc']}, precision_fake: {result['precision_fake']}, recall_fake: {result['recall_fake']}, f1_fake: {result['f1_fake']}, precision_real: {result['precision_real']}, recall_real: {result['recall_real']}, f1_real: {result['f1_real']}")




if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--data_file', type=str,
                        default='',
                        help='path to dataset annotations file')
    parser.add_argument('--data_dir', type=str,
                        default='',
                        help='dir to dataset annotations file')
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='testing batch size')
    parser.add_argument('--model_path', type=str,
                        default='',
                        help='path to the model')
    parser.add_argument('--metrics', action="store_true", help='calculate metrics')
    args = parser.parse_args()

    data_file = os.path.join(args.data_dir, f"{args.data_file}.json")
    dataset_name = data_file.split('/')[-1].split('.')[0]
    with open(data_file, 'r') as f:
        val_set = json.load(f)
    main(val_set, args, data_file)

