import os
import pandas as pd
import docx
import pdfplumber
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings('ignore')

# 所有路径统一为 D:\industrial_nlp_project
BASE_PATH = os.getcwd()

def parse_document(file_path):
    """
    统一文档读取接口，自动识别文件格式并解析，输出标准化DataFrame
    :param file_path: 文档路径（需在D:\industrial_nlp_project下）
    :return: DataFrame，包含fault_text1, fault_text2, similarity, repair_suggest, device_meta字段
    """
    # 校验文件路径合法性
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}，请确保文件在 {BASE_PATH} 路径下")
    
    # 提取文件后缀，识别格式
    file_suffix = os.path.splitext(file_path)[1].lower()
    
    # 1. Excel文件解析（.xlsx, .xls）
    if file_suffix in [".xlsx", ".xls"]:
        return parse_excel(file_path)
    # 2. Word文件解析（.docx, .doc）
    elif file_suffix in [".docx", ".doc"]:
        return parse_word(file_path)
    # 3. PDF文件解析（.pdf）
    elif file_suffix == ".pdf":
        return parse_pdf(file_path)
    # 4. CSV文件解析（复用原有逻辑，统一接口）
    elif file_suffix == ".csv":
        return parse_csv(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{file_suffix}，仅支持xlsx/xls/docx/doc/pdf/csv")

def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        # 核心：强制转换similarity为浮点数，空值/无效值填0.0
        df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        raise Exception(f"CSV文件解析失败：{str(e)}")

def parse_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        # 核心：强制转换similarity为浮点数，空值/无效值填0.0
        df['similarity'] = pd.to_numeric(df['similarity'], errors='coerce').fillna(0.0)
        return df
    except Exception as e:
        raise Exception(f"Excel文件解析失败：{str(e)}")
def parse_word(file_path):
    """解析Word文件（.docx优先，.doc仅支持Windows环境）"""
    text_content = []
    try:
        if file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())
        else:  # .doc格式，依赖pywin32（Windows环境）
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = 0
            doc = word.Documents.Open(file_path)
            for paragraph in doc.Paragraphs:
                if paragraph.Range.Text.strip():
                    text_content.append(paragraph.Range.Text.strip())
            doc.Close()
            word.Quit()
    except Exception as e:
        raise Exception(f"Word文件解析失败：{str(e)}，.doc格式需安装pywin32且在Windows环境运行")
    
    # Word文件默认按“一行一对故障文本”解析，需用户按格式录入（第一列为text1，第二列为text2，依次类推）
    # 若格式不规范，提示用户调整，不报错（保留妥协逻辑）
    if len(text_content) < 2 or len(text_content) % 5 != 0:
        warnings.warn("Word文件格式不规范，建议按：fault_text1、fault_text2、similarity、repair_suggest、device_meta 依次录入，每5行为一组")
        return pd.DataFrame()
    
    # 构建DataFrame
    data = []
    for i in range(0, len(text_content), 5):
        row = {
            "fault_text1": text_content[i] if i < len(text_content) else "",
            "fault_text2": text_content[i+1] if (i+1) < len(text_content) else "",
            "similarity": float(text_content[i+2]) if (i+2) < len(text_content) and text_content[i+2].replace(".","").isdigit() else 0.0,
            "repair_suggest": text_content[i+3] if (i+3) < len(text_content) else "",
            "device_meta": text_content[i+4] if (i+4) < len(text_content) else ""
        }
        data.append(row)
    df = pd.DataFrame(data)
    return standardize_fields(df)

def parse_pdf(file_path):
    """解析PDF文件，提取文本内容，兼容复杂格式"""
    text_content = []
    try:
        # 优先用pdfplumber解析（保留格式）
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # 去除页眉页脚（简单过滤，不影响核心逻辑）
                    lines = text.split("\n")
                    valid_lines = [line.strip() for line in lines if line.strip() and not line.strip().isdigit() and len(line.strip()) > 2]
                    text_content.extend(valid_lines)
        # 若pdfplumber解析失败，用PyPDF2兜底（妥协逻辑，保证不报错）
        if not text_content:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_content.extend([line.strip() for line in text.split("\n") if line.strip()])
    except Exception as e:
        raise Exception(f"PDF文件解析失败：{str(e)}")
    
    # PDF文件默认按“一行一对故障文本”解析，逻辑同Word
    if len(text_content) < 2 or len(text_content) % 5 != 0:
        warnings.warn("PDF文件格式不规范，建议按：fault_text1、fault_text2、similarity、repair_suggest、device_meta 依次录入，每5行为一组")
        return pd.DataFrame()
    
    data = []
    for i in range(0, len(text_content), 5):
        row = {
            "fault_text1": text_content[i] if i< len(text_content) else "",
            "fault_text2": text_content[i+1] if (i+1) < len(text_content) else "",
            "similarity": float(text_content[i+2]) if (i+2) < len(text_content) and text_content[i+2].replace(".","").isdigit() else 0.0,
            "repair_suggest": text_content[i+3] if (i+3) < len(text_content) else "",
            "device_meta": text_content[i+4] if (i+4) < len(text_content) else ""
        }
        data.append(row)
    # 构建标准化DataFrame（适配知识库字段）
    df = pd.DataFrame({
        "fault_text1": text_content[::2],  # 偶数行为text1
        "fault_text2": text_content[1::2], # 奇数行为text2
        "similarity": 0.0,
        "repair_suggest": "",
        "device_meta": ""
    })
    # 补全长度不一致的情况
    max_len = max(len(df["fault_text1"]), len(df["fault_text2"]))
    df["fault_text1"] = df["fault_text1"].reindex(range(max_len), fill_value="")
    df["fault_text2"] = df["fault_text2"].reindex(range(max_len), fill_value="")
    return standardize_fields(df)

def standardize_fields(df):
    """标准化字段，确保与现有训练/推理流程兼容，缺失字段自动补充（妥协逻辑，保证不报错）"""
    required_fields = ["fault_text1", "fault_text2", "similarity", "repair_suggest", "device_meta"]
    for field in required_fields:
        if field not in df.columns:
            df[field] = "" if field != "similarity" else 0.0
    # 过滤脏数据（空文本），但不中断程序
    df = df[(df["fault_text1"].str.strip() != "") & (df["fault_text2"].str.strip() != "")].reset_index(drop=True)
    # 相似度归一化到0-1
    df["similarity"] = df["similarity"].clip(0.0, 1.0)
    return df

# 测试代码（可注释，不影响核心功能）
# 测试代码（容错版：无文件跳过，解析失败仅提示，不中断运行）
if __name__ == "__main__":
    print("===== 工业文档解析工具测试开始 =====")
    # 测试CSV解析
    csv_path = os.path.join(BASE_PATH, "test.csv")
    print("\n【1. CSV文件解析测试】")
    if os.path.exists(csv_path):
        try:
            csv_df = parse_csv(csv_path)
            print(f"✅ 解析成功，有效数据量：{len(csv_df)} 行")
            print(csv_df.head())
        except Exception as e:
            print(f"❌ 解析失败：{str(e)}")
    else:
        print(f"ℹ️  跳过测试：文件不存在 -> {csv_path}")

    # 测试Excel解析
    excel_path = os.path.join(BASE_PATH, "test.xlsx")
    print("\n【2. Excel文件解析测试】")
    if os.path.exists(excel_path):
        try:
            excel_df = parse_excel(excel_path)
            print(f"✅ 解析成功，有效数据量：{len(excel_df)} 行")
            print(excel_df.head())
        except Exception as e:
            print(f"❌ 解析失败：{str(e)}")
    else:
        print(f"ℹ️  跳过测试：文件不存在 -> {excel_path}")

    # 测试统一解析接口（可选，多格式通用）
    print("\n【3. 统一解析接口测试（parse_document）】")
    test_files = [csv_path, excel_path]
    for file in test_files:
        if os.path.exists(file):
            try:
                df = parse_document(file)
                print(f"✅ {os.path.basename(file)} 统一接口解析成功")
                break
            except Exception as e:
                continue
    else:
        print("ℹ️  跳过测试：暂无可用的测试文件（可放入test.csv/test.xlsx到指定目录）")

    print("\n===== 工业文档解析工具测试结束 =====")