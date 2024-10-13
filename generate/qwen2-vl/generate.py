import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch

device = torch.device("cuda:2")

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/u2024001042/huggingface/Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto"
)
model = model.to(device)

# default processor
processor = AutoProcessor.from_pretrained(
    "/home/u2024001042/huggingface/Qwen/Qwen2-VL-7B-Instruct")

with open("data/testmini.json", "rb") as f:
    data = json.load(f)

output = []
batch_size = 5

# Process data in batches
for i in tqdm(range(0, len(data), batch_size), desc="Inferencing"):
    batch_data = data[i:i + batch_size]
    messages = []

    for datum in batch_data:
        img_question = datum["question"]
        img_option = datum["option"]
        img_path = datum["image_path"]

        prompt = f"Now, we require you to solve a multiple-choice math question.\n\
            Please briefly describe your thought process and provide the final answer(option).\n\
            Question: {img_question}\n\
            Option: {img_option}\n\
            Regarding the format, please answer following the template below, and be sure to include two <> symbols:\n\
            <Thought process>:<<your thought process>>\n\
            <Answer>:<<your option>>."

        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": "data/" + img_path},
            ]
        }]
        messages.append(message)

    # Preparation for batch inference
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Save results
    for datum, response in zip(batch_data, output_texts):
        datum["response"] = response
        output.append(datum)

with open("output/qwen2vl-voting.json", "w") as f:
    json.dump(output, f, indent=4)