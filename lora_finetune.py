import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import warnings
warnings.filterwarnings('ignore')

# 1. 加载本地Qwen-1.8B-Chat模型（CPU/GPU自动适配）
model_name = r"D:\deep\LlamaFactory-main\models\Qwen\Qwen1___5-1___8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>"
)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.float16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.float32
    )

# 2. LoRA轻量化配置（仅训练注意力层适配器）
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen专用模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 正常输出：trainable% 0.3%-0.5%

# 3. 加载工业故障微调数据
def load_finetune_data():
    import pandas as pd
    from datasets import Dataset
    data = pd.read_csv(r"D:\industrial_nlp_project\data\processed\train_processed.csv")
    finetune_data = []
    for _, row in data.iterrows():
        fault_text = row.get("fault_text1_clean", "").strip()
        solution = row.get("solution", "检查设备相关部件").strip()
        if fault_text:
            prompt = f"故障描述：{fault_text}\n维修方案：{solution}"
            finetune_data.append({"text": prompt})
    print(f"✅ 加载有效微调样本：{len(finetune_data)}条")
    return Dataset.from_list(finetune_data)

dataset = load_finetune_data()

# 4. 数据格式化（无维度混乱问题）
def format_data(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None,
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

dataset = dataset.map(format_data, batched=False)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5. 训练参数（核心修复：删除重复的gradient_accumulation_steps）
training_args = transformers.TrainingArguments(
    output_dir="./lora_finetuned_qwen",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 仅保留1次该参数
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    remove_unused_columns=True,
    dataloader_pin_memory=False,
    do_eval=False,
    fp16=torch.cuda.is_available(),  # GPU自动开启FP16，CPU关闭
    gradient_checkpointing=False,  # 禁用梯度检查点，避免报错
)

# 6. 启动训练+保存适配器
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
print(f"✅ 训练完成！LoRA适配器已保存至：{training_args.output_dir}")