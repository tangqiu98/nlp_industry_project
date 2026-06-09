from knowledge_base import FaultKnowledgeBase
import csv  # 用Python原生csv库替代pandas
import os

# 初始化知识库
kb = FaultKnowledgeBase(db_path="fault_knowledge_base.db")

# 定义数据文件路径（确保路径正确，若不存在会报错提示）
data_path = r"D:\卓面\实习项目\data\processed\train_preprocessed.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ 数据文件不存在：{data_path}\n请检查路径是否正确！")

# 用原生csv读取数据，手动处理字段填充
def load_csv_data(file_path):
    data = []
    with open(file_path, mode='r', encoding='utf-8-sig') as f:  # utf-8-sig处理BOM头
        reader = csv.DictReader(f)  # 按列名读取
        fieldnames = reader.fieldnames  # 获取所有列名
        
        for row in reader:
            # 提取核心字段，缺失则填充默认值
            fault_text = row.get("fault_text1_clean", "").strip()
            solution = row.get("solution", "").strip()
            similarity = float(row.get("similarity", 0.0))  # 默认为0.0
            repair_suggest = row.get("repair_suggest", "检查设备相关部件，排查故障原因").strip()
            device_meta = row.get("device_meta", "").strip()
            
            # 过滤空故障描述（仅保留有效数据）
            if fault_text:
                data.append({
                    "fault_text1_clean": fault_text,
                    "solution": solution,
                    "similarity": similarity,
                    "repair_suggest": repair_suggest,
                    "device_meta": device_meta
                })
    return data

# 加载并导入数据
print("🔄 正在加载数据...")
df_data = load_csv_data(data_path)
print(f"✅ 加载有效数据：{len(df_data)}条")

# 导入知识库（保持原逻辑不变）
kb.add_data(df_data, update_desc="初始化工业故障数据（原生Python实现）")

# 验证导入结果
print(f"✅ 知识库初始化完成，有效数据量：{kb.get_data_count()}")
all_faults = kb.get_all_faults()
if all_faults:
    print(f"✅ 示例故障数据：{all_faults[0]}")
else:
    print("⚠️  知识库中无故障数据，请检查数据文件是否有有效内容！")

# 关闭数据库连接
kb.close()
print("✅ 知识库连接已关闭，初始化完成！")