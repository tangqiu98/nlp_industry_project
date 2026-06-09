import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局变量+路径配置 =====================
BASE_DIR = r"D:\桌面\industrial_nlp_project (2)\industrial_nlp_project"
print(f"🔍 自动识别项目根目录：{BASE_DIR}")

# 数据路径
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DATA_PATH = os.path.join(DATA_PROCESSED_PATH, "train_processed.csv")
VAL_DATA_PATH = os.path.join(DATA_PROCESSED_PATH, "validation_processed.csv")
TEST_DATA_PATH = os.path.join(DATA_PROCESSED_PATH, "test_processed.csv")

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "bert_industrial")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "training_checkpoint.pth")

# 本地bert-base-chinese模型路径
BERT_MODEL_PATH = os.path.join(BASE_DIR, "model", "bert-base-chinese-local")

# 创建必要目录
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ===================== 系统验证 =====================
def verify_system():
    try:
        # 验证目录权限
        test_file = os.path.join(MODEL_SAVE_PATH, "_test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ 目录权限验证通过：{MODEL_SAVE_PATH}")

        # 验证bert-base-chinese模型
        if os.path.exists(BERT_MODEL_PATH):
            config_path = os.path.join(BERT_MODEL_PATH, "config.json")
            vocab_path = os.path.join(BERT_MODEL_PATH, "vocab.txt")
            if os.path.exists(config_path) and os.path.exists(vocab_path):
                print(f"✅ 检测到本地bert-base-chinese模型：{BERT_MODEL_PATH}")
            else:
                print(f"⚠️  bert-base-chinese模型文件不完整")
        else:
            print(f"⚠️  未检测到本地bert-base-chinese模型，请确保模型文件在：{BERT_MODEL_PATH}")

        return True
    except Exception as e:
        print(f"\n❌ 系统验证失败：{str(e)}")
        exit(1)

verify_system()

# ===================== 训练参数 =====================
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔧 训练配置：设备={DEVICE} | 批次大小={BATCH_SIZE} | 训练轮数={EPOCHS}")

# ===================== 导入transformers的BERT组件 =====================
try:
    from transformers import BertTokenizer, BertModel, BertConfig
    print("✅ transformers模块加载成功")
except ImportError:
    print("❌ 缺少transformers模块，正在安装...")
    os.system("pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
    from transformers import BertTokenizer, BertModel, BertConfig

# 加载本地bert-base-chinese的分词器
print(f"🔄 加载本地bert-base-chinese分词器...")
try:
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    print(f"✅ 分词器加载成功，词汇表大小：{tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ 分词器加载失败：{str(e)}")
    exit(1)

# ===================== BERT Siamese网络结构 =====================
class BertSiameseNetwork(nn.Module):
    """基于真实bert-base-chinese的Siamese网络"""
    def __init__(self, model_path=BERT_MODEL_PATH):
        super().__init__()
        # 加载预训练BERT
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # 相似度计算层：输入为[emb1, emb2, |emb1-emb2|]，维度为3*768
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def get_bert_embedding(self, input_ids, attention_mask):
        """获取BERT的[CLS] token embedding"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用pooler输出（[CLS] token经过线性层和tanh激活）
        return outputs.pooler_output.squeeze(-1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, meta_feature=None):
        # 获取两个文本的BERT嵌入
        embedding1 = self.get_bert_embedding(input_ids1, attention_mask1)
        embedding2 = self.get_bert_embedding(input_ids2, attention_mask2)

        # 计算嵌入差的绝对值
        diff = torch.abs(embedding1 - embedding2)

        # 拼接特征：[emb1, emb2, |emb1-emb2|]
        concat_feature = torch.cat([embedding1, embedding2, diff], dim=1)

        # 可选的元数据特征拼接
        if meta_feature is not None and meta_feature.shape[1] > 0:
            if meta_feature.shape[0] == concat_feature.shape[0]:
                concat_feature = torch.cat([concat_feature, meta_feature], dim=1)

        output = self.fc(concat_feature)
        return output

# 保留向后兼容的别名
SiameseSimilarityModel = BertSiameseNetwork

# ===================== 工具函数 =====================
def read_csv_safe(file_path):
    """安全读取CSV文件"""
    text1_list = []
    text2_list = []
    labels = []

    if not os.path.exists(file_path):
        print(f"⚠️  跳过不存在的文件：{file_path}")
        return text1_list, text2_list, labels

    try:
        import pandas as pd
        encodings = ["utf-8-sig", "gbk", "gb2312"]
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                break
            except:
                continue

        if df is None:
            print(f"⚠️  无法读取文件：{file_path}")
            return text1_list, text2_list, labels

        # 支持多种列名
        text1_cols = ["fault_text1_clean", "text1_clean", "text1", "故障文本1", "fault_text1"]
        text2_cols = ["fault_text2_clean", "text2_clean", "text2", "故障文本2", "fault_text2"]
        label_cols = ["semantic_similarity", "similarity", "相似度", "label"]

        t1_col = next((c for c in text1_cols if c in df.columns), None)
        t2_col = next((c for c in text2_cols if c in df.columns), None)
        label_col = next((c for c in label_cols if c in df.columns), None)

        if not all([t1_col, t2_col, label_col]):
            print(f"⚠️  文件{os.path.basename(file_path)}字段不完整，可用列：{list(df.columns)}")
            return text1_list, text2_list, labels

        text1_list = df[t1_col].fillna("").astype(str).tolist()
        text2_list = df[t2_col].fillna("").astype(str).tolist()
        labels = df[label_col].fillna(0.0).clip(0.0, 1.0).astype(float).tolist()

        # 过滤空文本
        valid_idx = [i for i, (t1, t2) in enumerate(zip(text1_list, text2_list)) if t1.strip() and t2.strip()]
        text1_list = [text1_list[i] for i in valid_idx]
        text2_list = [text2_list[i] for i in valid_idx]
        labels = [labels[i] for i in valid_idx]

        print(f"📄 成功读取{os.path.basename(file_path)}：{len(text1_list)}条样本")
        return text1_list, text2_list, labels

    except Exception as e:
        print(f"⚠️  读取文件失败：{str(e)}")
        return text1_list, text2_list, labels

# 自动安装pandas
try:
    import pandas as pd
except ImportError:
    print("❌ 缺少pandas，自动安装...")
    os.system("pip install pandas==2.2.2 -i https://pypi.tuna.tsinghua.edu.cn/simple")
    import pandas as pd

# ===================== 数据集类 =====================
class IndustrialDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64, is_train=True):
        self.text1, self.text2, self.labels = read_csv_safe(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t1 = self.text1[idx] if idx < len(self.text1) else ""
        t2 = self.text2[idx] if idx < len(self.text2) else ""
        label = self.labels[idx] if idx < len(self.labels) else 0.0

        # 使用bert-base-chinese分词器
        encoding1 = self.tokenizer(
            t1, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        encoding2 = self.tokenizer(
            t2, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )

        return {
            "input_ids1": encoding1["input_ids"].flatten(),
            "attention_mask1": encoding1["attention_mask"].flatten(),
            "input_ids2": encoding2["input_ids"].flatten(),
            "attention_mask2": encoding2["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.float32)
        }

# 保留向后兼容的别名
FaultDataset = IndustrialDataset

# ===================== 模型评估函数 =====================
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids1 = batch["input_ids1"].to(device)
            attention_mask1 = batch["attention_mask1"].to(device)
            input_ids2 = batch["input_ids2"].to(device)
            attention_mask2 = batch["attention_mask2"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)

            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input_ids1.size(0)

    return total_loss / len(dataloader.dataset)

# ===================== 训练函数 =====================
def train_model(train_data_path=None, val_data_path=None, tok=None, max_len=None, device=None, epochs=5, batch_size=2):
    """训练BERT Siamese相似度模型"""
    # 兼容默认参数
    train_data_path = train_data_path or TRAIN_DATA_PATH
    val_data_path = val_data_path or VAL_DATA_PATH
    tok = tok or tokenizer
    max_len = max_len or MAX_LEN
    device = device or DEVICE
    epochs = epochs or EPOCHS
    batch_size = batch_size or BATCH_SIZE

    print(f"\n🚀 启动bert-base-chinese Siamese模型训练")
    print(f"📁 模型保存路径：{MODEL_SAVE_PATH}")
    print(f"📁 BERT模型来源：{BERT_MODEL_PATH}")

    # 加载数据集
    train_dataset = IndustrialDataset(train_data_path, tok, max_len, is_train=True)
    val_dataset = IndustrialDataset(val_data_path, tok, max_len, is_train=False)
    test_dataset = IndustrialDataset(TEST_DATA_PATH, tok, max_len, is_train=False)

    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    print(f"\n✅ 数据加载完成：")
    print(f"  - 训练集：{len(train_dataset)}条 | 批次：{len(train_loader)}")
    print(f"  - 验证集：{len(val_dataset)}条 | 批次：{len(val_loader)}")
    print(f"  - 测试集：{len(test_dataset)}条 | 批次：{len(test_loader)}")

    # 初始化模型
    model = BertSiameseNetwork(model_path=BERT_MODEL_PATH).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_val_loss = float("inf")

    # 加载历史训练状态
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint.get('model_state_dict', {}), strict=False)
            optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            print(f"✅ 加载历史训练状态成功！从Epoch {start_epoch+1} 继续")
        except Exception as e:
            print(f"⚠️  历史状态加载失败：{str(e)}，从头训练")

    # 检查训练数据
    if len(train_loader) == 0:
        print("❌ 无有效训练数据，退出训练")
        return

    # 训练主循环
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        from tqdm import tqdm
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_bar:
            try:
                input_ids1 = batch["input_ids1"].to(device)
                attention_mask1 = batch["attention_mask1"].to(device)
                input_ids2 = batch["input_ids2"].to(device)
                attention_mask2 = batch["attention_mask2"].to(device)
                labels = batch["label"].to(device).float().unsqueeze(1)
            except Exception as e:
                print(f"⚠️  跳过异常批次：{str(e)}")
                continue

            optimizer.zero_grad()
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 验证阶段
        val_loss_avg = float("inf")
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for batch in val_bar:
                    try:
                        input_ids1 = batch["input_ids1"].to(device)
                        attention_mask1 = batch["attention_mask1"].to(device)
                        input_ids2 = batch["input_ids2"].to(device)
                        attention_mask2 = batch["attention_mask2"].to(device)
                        labels = batch["label"].to(device).float().unsqueeze(1)
                    except:
                        continue
                    outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            val_loss_avg = val_loss / len(val_loader)

        # 打印结果
        train_loss_avg = train_loss / len(train_loader)
        print(f"\n📊 Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        # 保存最优模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            try:
                os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
                save_path = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss
                }, save_path)
                print(f"✅ 最优模型已保存：{save_path} | 最优Val Loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"⚠️  最优模型保存失败：{str(e)}")
                torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_model_fallback.pth"))
                print(f"✅ 已降级保存到当前目录：best_model_fallback.pth")

        # 保存训练快照
        try:
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)
            print(f"✅ 训练快照已保存：{CHECKPOINT_PATH}\n")
        except Exception as e:
            print(f"⚠️  快照保存失败：{str(e)}")

    # 测试阶段
    print("=" * 50)
    if len(test_loader) > 0:
        print("开始测试模型...")
        try:
            checkpoint = torch.load(os.path.join(MODEL_SAVE_PATH, "best_model.pth"), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except:
            print("⚠️  加载最优模型失败，使用当前模型测试")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="[Test]")
            for batch in test_bar:
                try:
                    input_ids1 = batch["input_ids1"].to(device)
                    attention_mask1 = batch["attention_mask1"].to(device)
                    input_ids2 = batch["input_ids2"].to(device)
                    attention_mask2 = batch["attention_mask2"].to(device)
                    labels = batch["label"].to(device).float().unsqueeze(1)
                except:
                    continue
                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        test_loss_avg = test_loss / len(test_loader)
        print(f"\n📊 测试集Loss: {test_loss_avg:.4f}")

        try:
            with open(os.path.join(MODEL_SAVE_PATH, "test_result.txt"), "w", encoding="utf-8") as f:
                f.write(f"BERT-base-chinese Siamese相似度模型测试结果\n")
                f.write(f"BERT模型路径：{BERT_MODEL_PATH}\n")
                f.write(f"最优验证Loss: {best_val_loss:.4f}\n")
                f.write(f"测试Loss: {test_loss_avg:.4f}\n")
                f.write(f"训练轮数: {epochs}\n")
                f.write(f"批次大小: {batch_size}\n")
                f.write(f"最大序列长度: {max_len}\n")
            print("✅ 测试结果已保存")
        except:
            print("⚠️  测试结果保存失败")
    else:
        print("⚠️  无测试数据，跳过测试")

    print(f"\n🎉 训练完成！所有文件保存在：{MODEL_SAVE_PATH}")

# ===================== 知识库训练函数 =====================
def train_from_knowledge_base(knowledge_base, epochs=10, batch_size=32):
    """从知识库导出数据进行训练"""
    print("📌 从知识库导出数据进行训练...")
    train_data_path = knowledge_base.export_to_csv()

    if not os.path.exists(train_data_path):
        print("❌ 知识库数据导出失败，无法训练")
        return

    train_dataset = FaultDataset(train_data_path, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_data_path = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
    if os.path.exists(val_data_path):
        val_dataset = FaultDataset(val_data_path, tokenizer, MAX_LEN)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
        print("⚠️  未找到验证集，仅进行训练")

    # 初始化模型
    model = BertSiameseNetwork(model_path=BERT_MODEL_PATH).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        from tqdm import tqdm
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids1 = batch["input_ids1"].to(DEVICE)
            attention_mask1 = batch["attention_mask1"].to(DEVICE)
            input_ids2 = batch["input_ids2"].to(DEVICE)
            attention_mask2 = batch["attention_mask2"].to(DEVICE)
            labels = batch["label"].to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids1.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} | 训练损失：{avg_train_loss:.4f}")

        # 验证
        if val_loader is not None:
            val_loss = evaluate_model(model, val_loader, criterion, DEVICE)
            print(f"Epoch {epoch+1} | 验证损失：{val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(BASE_DIR, "best_model_fallback.pth"))
                print(f"✅ 保存最优模型（验证损失：{best_val_loss:.4f}）")

        # 保存训练快照
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(BASE_DIR, f"checkpoint_epoch_{epoch+1}_fallback.pth"))
            print(f"✅ 保存第{epoch+1}轮训练快照")

    print("🎉 从知识库导出数据训练完成！")

# ===================== 主函数 =====================
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 64

    # 知识库训练入口（可选）
    use_knowledge_base = False
    if use_knowledge_base:
        from knowledge_base import FaultKnowledgeBase
        kb = FaultKnowledgeBase()
        train_from_knowledge_base(kb, epochs=10, batch_size=32)
        kb.close()
    else:
        # 使用处理好的数据训练
        train_data_path = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
        val_data_path = os.path.join(BASE_DIR, "data", "processed", "validation_processed.csv")

        if os.path.exists(train_data_path):
            train_model(train_data_path, val_data_path, tokenizer, MAX_LEN, DEVICE, epochs=10, batch_size=8)
        else:
            print(f"❌ 训练数据不存在：{train_data_path}")
            print("请先准备好训练数据")