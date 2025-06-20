import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import json
import re
import os
import torch
import numpy as np
from datasets import Dataset
from rouge import Rouge
from tqdm import tqdm
from difflib import SequenceMatcher
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# ----------------------------
# 系统提示和XML格式定义
# ----------------------------
SYSTEM_PROMPT = """
你是一个专业的学术引用检测助手。请分析给定的两个文本片段，判断x1是否正确引用了x2。
请严格按照以下格式回答：
<reasoning>
...你的推理过程...
</reasoning>
<answer>
...你的最终答案（只能是yes或no）...
</answer>
"""

XML_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# ----------------------------
# 数据加载与处理
# ----------------------------
def load_jsonl_data(file_path):
    """加载JSONL格式的数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_citation_questions(file_path):
    """创建文献引用数据集，使用对话格式"""
    data = load_jsonl_data(file_path)
    
    formatted_data = []
    for item in data:
        x1 = item['x1']
        x2 = item['x2']
        label = item['label']
        
        # 构建用户提示
        user_prompt = f"x1: {x1}\nx2: {x2}\n请判断x1是否正确引用了x2。"
        
        # 构建参考答案
        answer = XML_FORMAT.format(
            reasoning=label['reason'],
            answer=label['answer']
        )
        
        formatted_data.append({
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            'answer': answer
        })
    
    return formatted_data

# ----------------------------
# XML提取函数
# ----------------------------
def extract_xml_reasoning(text: str) -> str:
    """从文本中提取<reasoning>标签内容"""
    try:
        start = text.find("<reasoning>") + len("<reasoning>")
        end = text.find("</reasoning>")
        if start != -1 and end != -1:
            return text[start:end].strip()
    except:
        pass
    return ""

def extract_xml_answer(text: str) -> str:
    """从文本中提取<answer>标签内容"""
    try:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>")
        if start != -1 and end != -1:
            return text[start:end].strip().lower()
    except:
        pass
    return ""

# ----------------------------
# 相似度计算
# ----------------------------
def similarity_score(text1, text2):
    """计算两个文本之间的相似度得分"""
    if not text1 or not text2:
        return 0.0
    try:
        # 使用ROUGE-L
        rouge = Rouge()
        scores = rouge.get_scores(text1, text2)
        return scores[0]['rouge-l']['f']
    except:
        # 备选方案：序列匹配
        return SequenceMatcher(None, text1, text2).ratio()

# ----------------------------
# 模型评估
# ----------------------------
def evaluate_model(model, tokenizer, dataset, max_length=1024):
    """评估模型在验证集上的性能"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    reasoning_scores = []
    
    for example in tqdm(dataset):
        # 构建对话格式的输入
        messages = example['prompt']
        prompt_text = ""
        for msg in messages:
            if msg['role'] == 'system':
                prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'user':
                prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
        
        # 准备输入
        inputs = tokenizer(prompt_text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案和推理
        model_answer = extract_xml_answer(response)
        model_reasoning = extract_xml_reasoning(response)
        ref_answer = extract_xml_answer(example['answer'])
        ref_reasoning = extract_xml_reasoning(example['answer'])
        
        # 计算准确率
        if model_answer == ref_answer:
            correct_predictions += 1
        total_predictions += 1
        
        # 计算推理质量
        if model_reasoning and ref_reasoning:
            sim_score = similarity_score(model_reasoning, ref_reasoning)
            reasoning_scores.append(sim_score)
    
    # 计算指标
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_reasoning = np.mean(reasoning_scores) if reasoning_scores else 0
    
    return {
        "accuracy": accuracy,
        "avg_reasoning_score": avg_reasoning,
        "num_examples": total_predictions,
    }

# ----------------------------
# 主函数 - 使用Modelscope加载Qwen2.5-3B模型
# ----------------------------
def main():
    # 配置参数
    data_path = "/home/wuxiao/test/citation_agent_app/citation_data.jsonl"
    model_name = "qwen/Qwen2.5-3B-Instruct"  # Modelscope模型ID
    local_model_dir = "/home/wuxiao/test/citation_agent_app/models/Qwen/Qwen2.5-3B-Instruct"  # 本地模型目录
    output_dir = "fine_tuned_qwen2.5_3B"
    max_seq_length = 2048  # 增加序列长度以适应更大模型
    lora_rank = 32  # 增加LoRA秩以提升模型容量
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    formatted_data = get_citation_questions(data_path)
    
    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        formatted_data, test_size=0.2, random_state=42
    )
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 加载模型和分词器 - Qwen2.5-3B
    print("加载Qwen2.5-3B模型和分词器...")
    
    # 检查本地是否有模型，如果没有则从Modelscope下载
    if not os.path.exists(local_model_dir):
        print(f"本地模型不存在，从Modelscope下载: {model_name}")
        model_dir = snapshot_download(model_name, cache_dir=local_model_dir)
    else:
        model_dir = local_model_dir
        print(f"使用本地模型: {model_dir}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )
    
    # 设置特殊token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # 应用LoRA适配器
    print("应用LoRA适配器...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2*lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备数据集处理函数
    def preprocess_function(examples):
        # 构建对话格式输入
        prompts = []
        for messages in examples["prompt"]:
            prompt_text = ""
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'user':
                    prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            prompt_text += "<|im_start|>assistant\n"
            prompts.append(prompt_text)
        
        # 分词
        model_inputs = tokenizer(
            prompts,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        
        # 添加答案字段
        model_inputs["labels"] = tokenizer(
            [answer for answer in examples["answer"]],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )["input_ids"]
        
        return model_inputs
    
    # 预处理数据集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # 配置训练参数 - 修复参数名错误
    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # 减少批次大小以适应更大模型
        gradient_accumulation_steps=8,   # 增加梯度累积步数
        learning_rate=1e-5,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=100,
        # 修复参数名错误
        eval_strategy="steps",  # 使用新的参数名
        eval_steps=100,         # 使用新的参数名
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=2,
    )
    # from trl import GRPOConfig, GRPOTrainer
    # training_args = GRPOConfig(
    #     learning_rate = 5e-6,
    #     adam_beta1 = 0.9,
    #     adam_beta2 = 0.99,
    #     weight_decay = 0.1,
    #     warmup_ratio = 0.1,
    #     lr_scheduler_type = "cosine",
    #     optim = "paged_adamw_8bit",
    #     logging_steps = 1,
    #     per_device_train_batch_size = 1,
    #     gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    #     num_generations = 6, # Decrease if out of memory
    #     max_prompt_length = max_prompt_length,
    #     max_completion_length = max_seq_length - max_prompt_length,
    #     num_train_epochs = 1, # Set to 1 for a full training run
    #     max_steps = 250,
    #     save_steps = 250,
    #     max_grad_norm = 0.1,
    #     report_to = "none", # Can use Weights & Biases
    #     output_dir = "outputs",
    # )
    
    # 初始化训练器
    print("初始化训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    # trainer = GRPOTrainer(
    # model = model,
    # processing_class = tokenizer,
    # reward_funcs = [
    #     xmlcount_reward_func,
    #     soft_format_reward_func,
    #     strict_format_reward_func,
    #     int_reward_func,#有合适的奖励函数可以放进去
    #     correctness_reward_func,
    # ],
    # args = training_args,
    # train_dataset = dataset,
    # )
    
    # 评估微调前的模型
    print("评估微调前的模型...")
    pretrained_metrics = evaluate_model(model, tokenizer, val_data)
    print(f"微调前准确率: {pretrained_metrics['accuracy']:.4f}")
    print(f"微调前平均推理质量: {pretrained_metrics['avg_reasoning_score']:.4f}")
    
    # 训练模型
    print("开始训练...")
    trainer.train()
    
    # 保存微调后的模型
    print("保存微调后的模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 评估微调后的模型
    print("评估微调后的模型...")
    fine_tuned_metrics = evaluate_model(model, tokenizer, val_data)
    print(f"微调后准确率: {fine_tuned_metrics['accuracy']:.4f}")
    print(f"微调后平均推理质量: {fine_tuned_metrics['avg_reasoning_score']:.4f}")
    
    # 推理示例
    print("\n推理示例:")
    x1 = "Prior research on graph neural networks emphasized message passing mechanisms. As demonstrated in Lee & Park (2023), 'adaptive attention gates reduced over-smoothing by 37% in deep GNN architectures'."
    x2 = "Our proposed adaptive gating mechanism suppresses feature homogenization, decreasing over-smoothing by 37.2% in 10-layer GCNs (Table 3)."
    
    # 构建对话格式输入
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f"x1: {x1}\nx2: {x2}\n请判断x1是否正确引用了x2。"}
    ]
    
    prompt_text = ""
    for msg in messages:
        if msg['role'] == 'system':
            prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg['role'] == 'user':
            prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
    prompt_text += "<|im_start|>assistant\n"
    
    # 准备输入
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"完整输出:\n{response}")
    print(f"提取的推理: {extract_xml_reasoning(response)}")
    print(f"提取的答案: {extract_xml_answer(response)}")

if __name__ == "__main__":
    main()