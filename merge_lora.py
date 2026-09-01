import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 配置路径 ---
# 1. 基础模型路径 
base_model_path = "./llm_weights/llama-2-7b" 

# 2. 训练好的 LoRA 路径 
lora_path = "./models/attack/advprompter_trained"

# 3. 合并后的保存路径 
output_path = "./models/attack/advprompter_merged"

print(f"正在加载基础模型: {base_model_path} ...")
# 加载基础模型 
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"正在加载 LoRA 权重: {lora_path} ...")
# 加载 LoRA
model = PeftModel.from_pretrained(base_model, lora_path)

print("正在合并权重 (Merge and Unload)...")
# 关键步骤：把 LoRA 权重这一层“贴”死到基础模型上
model = model.merge_and_unload()

print(f"正在保存完整模型到: {output_path} ...")
# 保存模型和 Tokenizer
model.save_pretrained(output_path)

# 把 Tokenizer 也搬过去，否则 vllm 没法把字转成数字
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("✅ 合并完成！")