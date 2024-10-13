import json
import re
import torch
from collections import Counter
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

device = torch.device("cuda:3")
print("Using device:", device)

# Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/u2024001042/huggingface/Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto"
)
model = model.to(device)

# Default processor
processor = AutoProcessor.from_pretrained("/home/u2024001042/huggingface/Qwen/Qwen2-VL-7B-Instruct")

with open("data/testmini.json", "rb") as f:
    data = json.load(f)

output = []
width = 5

for datum in tqdm(data, desc="Inferencing"):
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

    responses = []
    for i in range(width):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": "data/" + img_path},
            ],
        }]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )    
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)


        generated_ids = model.generate(**inputs, max_new_tokens=1024, temperature=1, top_k=50, top_p=0.95)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        responses.append(response[0])

    datum["all_responses"] = responses

    options = []
    for response in responses:
        option = response.split('Answer')[-1].strip()
        option = re.sub(r'[>><<:.]', '', option).strip()
        option = option[0] if option and option[0] in 'ABCDEFGH' else None
        options.append(option)

    datum["options"] = options

    option_response_map = {option: response for option, response in zip(options, responses)}
    most_common_option = Counter(options).most_common(1)[0][0]

    final_response = option_response_map.get(most_common_option)

    datum["response"] = final_response
    output.append(datum)

with open("output/qwen2vl-voting.json", "w") as f:
    json.dump(output, f, indent=4)