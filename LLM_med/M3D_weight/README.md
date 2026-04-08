---
license: apache-2.0
---

# M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models

[**Paper**](https://arxiv.org/abs/2404.00578) | [**Data**](https://github.com/BAAI-DCAI/M3D?tab=readme-ov-file#data) | [**Code**](https://github.com/BAAI-DCAI/M3D)

M3D is the pioneering and comprehensive series of work on the  multi-modal large language model for 3D medical analysis, including:
- **M3D-Data**: the largest-scale open-source 3D medical dataset, consists of 120K image-text pairs and 662K instruction-response pairs;
- **M3D-LaMed**: the versatile multi-modal models with M3D-CLIP pretrained vision encoder, which are capable of tasks such as image-text retrieval, report generation, visual question answering, positioning and segmentation;
- **M3D-Bench**: the most comprehensive automatic evaluation benchmark covers 8 tasks.

## Notifications
- We found that the previous GoodBaiBai88/M3D-LaMed-Llama-2-7B model had problems in the segmentation task. We have fixed this problem and will re-release the new model in the next few days.

## Quickstart
Here, we can easily use our model based on Hugging Face.

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = 'GoodBaiBai88/M3D-LaMed-Llama-2-7B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy 
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "./Data/data/examples/example_01.npy"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

model = model.to(device=device)

# question = "Can you provide a caption consists of findings for this medical image?"
question = "What is liver in this image? Please output the segmentation mask."
# question = "What is liver in this image? Please output the box."

image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

print('question', question)
print('generated_texts', generated_texts[0])

image = sikt.GetImageFromArray(image_np)
ssv.display(image)
seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
ssv.display(seg)
```