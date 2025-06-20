import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# ✅ 模型本地路径（替换为你下载的路径）
MODEL_DIR = "/home/wuxiao/test/citation_agent_app/internlm-chat-7b/Shanghai_AI_Laboratory/internlm-chat-7b"

# ✅ 加载 tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

# ✅ LoRA 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)

# ✅ 加载数据
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return [{
        "text": f"""### 上下文:
{x["x1"]}

### 引文:
{x["x2"]}

### 判断:
{x["label"]["answer"]}。原因：{x["label"]["reason"]}"""
    } for x in data]

# ✅ Tokenizer
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

data = load_data("/home/wuxiao/test/citation_agent_app/citation_data.jsonl")
dataset = Dataset.from_list(data).train_test_split(test_size=0.1)
train_dataset = dataset["train"].map(tokenize)
eval_dataset = dataset["test"].map(tokenize)

# ✅ 训练参数 (修复参数名)
training_args = TrainingArguments(
    output_dir="./sft_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    # 修改这里 ↓
    eval_strategy="epoch",  # 使用新参数名 eval_strategy 替代 evaluation_strategy
    save_strategy="epoch",
    # 修改这里 ↑
    save_total_limit=2,
    report_to="none",
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ✅ 开始训练
trainer.train()
trainer.save_model("./sft_output")
tokenizer.save_pretrained("./sft_output")

