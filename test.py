from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("/fs/archive/share/u2024001021/huggingface_models/llama3-llava-next-8b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("/fs/archive/share/u2024001021/huggingface_models/llama3-llava-next-8b-hf", torch_dtype=torch.float16, device_map="auto") 

# prepare image and text prompt, using the appropriate prompt template
url = "http://gips2.baidu.com/it/u=195724436,3554684702&fm=3028&app=3028&f=JPEG&fmt=auto?w=1280&h=960"
image = Image.open("data/2steps/image/1-1.png")
print("done")

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What is shown in this image?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
