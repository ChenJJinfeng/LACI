import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = '/data/chenjinfeng/code/VL/pre_weight/M3D' # 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "/data/chenjinfeng/Data/CT_160/Xinan/image_CT_npy/1_d.npy"

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

# Freezing model parameters
for param in model.parameters():
    param.requires_grad = False
    # print()

for n, p in model.named_parameters():       # 打印所有参数名称
    print(n)

for name, module in model.named_modules():      # 打印所有模块名称
    print(name)





features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
# 添加多个层的钩子
layers_to_hook = [
    'model.vision_tower.vision_tower.blocks.4.norm2',
    'model.vision_tower.vision_tower.blocks.5.norm2'
]

for layer_name in layers_to_hook:
    layer = dict([*model.named_modules()])[layer_name]
    layer.register_forward_hook(get_features(layer_name))

# 运行前向传播
# output = model(x)

# 获取特征
v_proj_features = features['vision_tower.vision_tower.blocks.4.norm2']
gate_proj_features = features['vision_tower.vision_tower.blocks.5.norm2']

print(v_proj_features.shape, gate_proj_features.shape)

# class CustomLlamaModel(torch.nn.Module):
#     def __init__(self, original_model, layers_to_extract):
#         super().__init__()
#         self.original_model = original_model
#         self.layers_to_extract = layers_to_extract
# 
#     def forward(self, input_ids, image_tensor):
#         hidden_states = self.original_model.model.embed_tokens(input_ids)
#         all_hidden_states = []
# 
#         for i, layer in enumerate(self.original_model.model.layers):
#             print(f"Layer {i+1}: {layer}")
#             hidden_states, _, _ = layer(hidden_states)
#             if i in self.layers_to_extract:
#                 all_hidden_states.append(hidden_states)
# 
#         # Generate text and segmentation mask
#         generation, seg_logit = self.original_model.generate(
#             image_tensor, input_ids, seg_enable=True,
#             max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0
#         )
# 
#         return generation, seg_logit, all_hidden_states
# 
# # Wrap the original model
# layers_to_extract = [0, 6, 13, 27]  # Adjusted for zero-based indexing
# custom_model = CustomLlamaModel(model, layers_to_extract)
# 
# # Example input
# question = "What is liver in this image? Please output the segmentation mask."
# image_tokens = "<im_patch>" * proj_out_num
# input_txt = image_tokens + question
# input_ids = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
# 
# image_np = np.load(image_path)
# image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)
# 
# # Forward pass to extract features and generate text and mask
# generation, seg_logit, extracted_features = custom_model(input_ids, image_pt)
# 
# # generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
# # seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
# # print('Question:', question)
# # print('Generated Text:', generated_texts[0])
# #
# # # Display image and segmentation mask
# # image = sikt.GetImageFromArray(image_np)
# # ssv.display(image)
# # seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
# # ssv.display(seg)
# 
# # Display features from specified layers
# for idx, features in enumerate(extracted_features):
#     print(f"Features from layer {layers_to_extract[idx] + 1}: {features.shape}")

