
# 下载并测试 InternLM-chat-7B 模型
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_PROMPT = "你是一个科研助手，判断某个引用是否准确，并输出 answer（yes/no）和 reason（引用是否正确的理由）格式如下：<xml><answer>...</answer><reason>...</reason></xml>。"
x1 = "Limitations of the state of the art The conventional locks [3, 5, 9, 10, 14] oftentimes tend to equalize the number of lock acquisitions by adhering to either random or pseudo-FIFO policy, while ignoring various characteristics of lock usage, such as the CS lengths and the frequencies of lock acquisition requests. To overcome the aforementioned problem, [12] proposed a scheduler-cooperative mutex, called SCL. SCL monitors lock usage and forcibly suspends certain threads if they hinder CPU time fairness."
x2 = "SMURF-THP: Score Matching-based UnceRtainty quantiFication for Transformer Hawkes Process Table 5. Event type prediction accuracy of SMURF-THP given ground truth event time as the inputs. We also train SMURF-THP with different volumes of training data to study its generalization ability. We train the model on different ratios of the dataset and present the performance in Figure 6 and Figure 7."
messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': f"x1: {x1}\nx2: {x2}\n请判断x1是否正确引用了x2。"}
]

# 构建 prompt
prompt_text = ""
for msg in messages:
    if msg['role'] == 'system':
        prompt_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
    elif msg['role'] == 'user':
        prompt_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
prompt_text += "<|im_start|>assistant\n"

# 推理函数
def generate_response(model_path, prompt_text, title=""):
    print(f"\n===== 模型：{title} =====")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
    print(f"输出结果：\n{response}")
    return response


# 跑微调后的模型
generate_response("/home/wuxiao/test/citation_agent/fine_tuned_qwen2.5_3B", prompt_text, title="微调后模型")

 