import pandas as pd
import json
import re

# 跨企业工业术语归一化词典（可根据业务扩展）
GLOBAL_INDUSTRIAL_TERM = {
    # 设备类
    "油缸": "液压缸", "油压缸": "液压缸", "液压筒": "液压缸",
    "缓冲器": "缓冲装置", "减震器": "缓冲装置",
    # 故障类
    "渗漏": "泄漏", "滴漏": "泄漏", "异音": "异响",
    "卡滞": "卡死", "抱死": "卡死", "磨耗": "磨损",
    # 动作类
    "停机": "故障停机", "跳闸": "故障停机", "过载": "超负荷"
}

def parse_industrial_ticket(file_path):
    """多格式工单解析：CSV/Excel/JSON，解决跨企业数据异构"""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("仅支持CSV/Excel/JSON格式的工单文件")
    return df

def normalize_industrial_term(text):
    """工业术语归一化，统一跨企业术语"""
    if not isinstance(text, str) or text == "":
        return ""
    # 替换归一化术语
    for old, new in GLOBAL_INDUSTRIAL_TERM.items():
        text = re.sub(r'\b' + old + r'\b', new, text)
    # 去除特殊字符，清洗文本
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_industrial_data(file_path, save_path):
    """完整工业数据预处理流程：解析→归一化→清洗→保存"""
    # 1. 多格式解析
    df = parse_industrial_ticket(file_path)
    # 2. 检查核心列，自动重命名（适配不同企业的列名）
    col_mapping = {}
    for col in df.columns:
        if "故障文本1" in col or "text1" in col.lower():
            col_mapping[col] = "fault_text1_clean"
        elif "故障文本2" in col or "text2" in col.lower():
            col_mapping[col] = "fault_text2_clean"
    df.rename(columns=col_mapping, inplace=True)
    # 3. 术语归一化
    df["fault_text1_clean"] = df["fault_text1_clean"].apply(normalize_industrial_term)
    df["fault_text2_clean"] = df["fault_text2_clean"].apply(normalize_industrial_term)
    # 4. 保存预处理后的数据（统一为CSV格式）
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 跨企业数据预处理完成，保存至：{save_path}")
    print(f"📊 处理后数据量：{len(df)} 条 | 术语已统一为工业标准术语")
    return df

if __name__ == "__main__":
    # 输入你真实存在的训练数据
    INPUT_FILE = r"D:\卓面\实习项目\data\raw\fault_semantic_matching_train.csv"
    # 输出到你已有的 processed 目录
    OUTPUT_FILE = r"D:\卓面\实习项目\data\processed\train_preprocessed.csv"
    preprocess_industrial_data(INPUT_FILE, OUTPUT_FILE)