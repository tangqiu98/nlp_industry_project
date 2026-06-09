import torch
import torch.nn as nn
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== 路径配置 =====================
BASE_DIR = r"D:\桌面\industrial_nlp_project (2)\industrial_nlp_project"
BERT_MODEL_PATH = os.path.join(BASE_DIR, "model", "bert-base-chinese-local")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "bert_industrial")

# 模型路径
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "best_model_fallback.pth")
LIGHT_MODEL_PATH = os.path.join(BASE_DIR, "quantized_pruned_model.pth")

# 固定参数
MAX_LEN = 64
DEVICE = torch.device("cpu")
PRUNE_RATIO = 0.3
QUANTIZE_DTYPE = torch.qint8

print(f"🔧 轻量化配置：剪枝比例=30% | 量化=INT8")
print(f"✅ 输入模型：{BEST_MODEL_PATH}")
print(f"✅ 输出模型：{LIGHT_MODEL_PATH}")

# ===================== 导入transformers组件 =====================
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("❌ 缺少transformers模块，正在安装...")
    os.system("pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
    from transformers import BertTokenizer, BertModel

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
print(f"✅ 分词器加载成功：{BERT_MODEL_PATH}")

# ===================== BERT Siamese网络结构 =====================
class BertSiameseNetwork(nn.Module):
    """基于bert-base-chinese的Siamese网络（与bert.py保持一致）"""
    def __init__(self, model_path=BERT_MODEL_PATH):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size  # 768

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output.squeeze(-1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, meta_feature=None):
        embedding1 = self.get_bert_embedding(input_ids1, attention_mask1)
        embedding2 = self.get_bert_embedding(input_ids2, attention_mask2)
        diff = torch.abs(embedding1 - embedding2)
        concat_feature = torch.cat([embedding1, embedding2, diff], dim=1)

        if meta_feature is not None and meta_feature.shape[1] > 0:
            if meta_feature.shape[0] == concat_feature.shape[0]:
                concat_feature = torch.cat([concat_feature, meta_feature], dim=1)

        output = self.fc(concat_feature)
        return output

# 向后兼容别名
SiameseSimilarityModel = BertSiameseNetwork

# ===================== 纯PyTorch剪枝 =====================
def prune_model(model):
    """L1范数剪枝，移除小权重"""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            weight = m.weight.data.clone()
            l1 = torch.abs(weight).flatten()
            k = int(PRUNE_RATIO * l1.numel())
            if k > 0:
                thres = torch.kthvalue(l1, k).values
                m.weight.data = weight * (torch.abs(weight) > thres).float()
    print("✅ 模型剪枝完成")
    return model

# ===================== PyTorch量化 =====================
def quantize_model(model):
    """动态量化到INT8"""
    model.eval()
    quant_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=QUANTIZE_DTYPE
    )
    print("✅ 模型INT8量化完成")
    return quant_model

# ===================== 多算法相似度计算 =====================
from similarity_calculator import MultiSimilarityCalculator

def verify_with_similarity(model, tokenizer):
    """使用多算法验证轻量化模型"""
    print("\n📊 多算法相似度验证")
    calculator = MultiSimilarityCalculator()

    # 工业故障文本示例
    text1 = "液压缸泄漏，无法正常加压"
    text2 = "油缸渗漏，压力无法维持"

    # BERT推理
    with torch.no_grad():
        encoding1 = tokenizer(text1, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        encoding2 = tokenizer(text2, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        output = model(encoding1["input_ids"], encoding1["attention_mask"],
                       encoding2["input_ids"], encoding2["attention_mask"])
        bert_sim = round(torch.clip(output, 0.0, 1.0).cpu().item(), 4)

    # 多算法对比
    multi_sim = calculator.print_similarity_comparison(text1, text2, "", "", model)

    print(f"  BERT相似度: {bert_sim}")
    print(f"  融合相似度: {multi_sim.get('fused_similarity', 'N/A')}")

# ===================== 保存并验证 =====================
def save_and_verify_light_model(model, save_path):
    light_model_path = save_path
    torch.save(model.state_dict(), light_model_path)

    # 计算模型大小
    original_model_size = os.path.getsize(BEST_MODEL_PATH) / 1024 / 1024 if os.path.exists(BEST_MODEL_PATH) else 0
    light_model_size = os.path.getsize(light_model_path) / 1024 / 1024

    print(f"\n📊 模型体积对比：")
    if original_model_size > 0:
        size_reduce_ratio = (1 - light_model_size / original_model_size) * 100
        print(f"   原模型：{original_model_size:.2f} MB")
    print(f"   轻量化模型：{light_model_size:.2f} MB")
    if original_model_size > 0:
        print(f"   体积缩小：{size_reduce_ratio:.1f}%")
    print(f"✅ 轻量化模型已保存：{light_model_path}")

    # 验证模型
    verify_with_similarity(model, tokenizer)

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1. 加载原始模型
    model = BertSiameseNetwork(model_path=BERT_MODEL_PATH)

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE), strict=False)
        print(f"✅ 原始模型加载完成：{BEST_MODEL_PATH}")
    elif os.path.exists(FALLBACK_MODEL_PATH):
        model.load_state_dict(torch.load(FALLBACK_MODEL_PATH, map_location=DEVICE), strict=False)
        print(f"✅ 原始模型加载完成（fallback）：{FALLBACK_MODEL_PATH}")
    else:
        print("❌ 未找到训练模型，请先运行bert.py进行训练")
        exit(1)

    # 2. 剪枝
    model = prune_model(model)

    # 3. 量化
    model = quantize_model(model)

    # 4. 保存轻量化模型
    save_and_verify_light_model(model, LIGHT_MODEL_PATH)