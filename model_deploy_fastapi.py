import os
import torch
import json
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Any
warnings.filterwarnings('ignore')

# ========== 核心配置 ==========
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
BASE_DIR = r"D:\桌面\industrial_nlp_project (2)\industrial_nlp_project"
BERT_MODEL_PATH = os.path.join(BASE_DIR, "model", "bert-base-chinese-local")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "bert_industrial")

LIGHT_MODEL_PATH = os.path.join(BASE_DIR, "quantized_pruned_model.pth")
BACKUP_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "best_model.pth")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "best_model_fallback.pth")

MAX_LEN = 64
DEVICE = torch.device("cpu")
APP_HOST = "0.0.0.0"
APP_PORT = 8000

# ========== 导入transformers组件 ==========
try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    print("❌ 缺少transformers模块，正在安装...")
    import subprocess
    subprocess.run(["pip", "install", "transformers", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    from transformers import BertTokenizer, BertModel

# 加载bert-base-chinese分词器
print(f"🔄 加载bert-base-chinese分词器...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
print(f"✅ 分词器加载成功")

# ========== 导入项目内部模块 ==========
try:
    from similarity_calculator import MultiSimilarityCalculator
    from knowledge_base import FaultKnowledgeBase
    from suggestion_generator import RepairSuggestionGenerator
    from rag_engine import IndustrialFaultRAGEngine
    from langchain_community.llms import Ollama
except ImportError as e:
    print(f"⚠️  部分模块导入失败：{str(e)}，将使用降级方案")

# ========== BERT Siamese网络结构（与bert.py保持一致） ==========
class BertSiameseNetwork(torch.nn.Module):
    def __init__(self, model_path=BERT_MODEL_PATH):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def get_bert_embedding(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output.squeeze(-1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        embedding1 = self.get_bert_embedding(input_ids1, attention_mask1)
        embedding2 = self.get_bert_embedding(input_ids2, attention_mask2)
        diff = torch.abs(embedding1 - embedding2)
        concat_feature = torch.cat([embedding1, embedding2, diff], dim=1)
        return self.fc(concat_feature)

# 向后兼容别名
SiameseSimilarityModel = BertSiameseNetwork

# ========== 初始化组件 ==========
print(f"🔧 部署配置：设备={DEVICE} | 模型路径：{LIGHT_MODEL_PATH}")

print("\n🚀 加载模型和核心组件...")

# 加载模型
model = BertSiameseNetwork(model_path=BERT_MODEL_PATH)

if os.path.exists(LIGHT_MODEL_PATH):
    model.load_state_dict(torch.load(LIGHT_MODEL_PATH, map_location=DEVICE), strict=False)
    print(f"✅ 加载轻量化模型成功：{LIGHT_MODEL_PATH}")
elif os.path.exists(BACKUP_MODEL_PATH):
    model.load_state_dict(torch.load(BACKUP_MODEL_PATH, map_location=DEVICE), strict=False)
    print(f"✅ 加载备份模型成功：{BACKUP_MODEL_PATH}")
elif os.path.exists(FALLBACK_MODEL_PATH):
    model.load_state_dict(torch.load(FALLBACK_MODEL_PATH, map_location=DEVICE), strict=False)
    print(f"✅ 加载fallback模型成功：{FALLBACK_MODEL_PATH}")
else:
    print("⚠️  未找到训练模型，将使用未训练的BERT模型")

model.eval()

# 初始化项目组件
try:
    knowledge_base = FaultKnowledgeBase()
    similarity_calculator = MultiSimilarityCalculator()
    suggestion_generator = RepairSuggestionGenerator(knowledge_base)
    rag_engine = IndustrialFaultRAGEngine(knowledge_base=knowledge_base, lora_adapter_path="./lora_finetuned_qwen")
    llm = Ollama(model="qwen:0.5b")
    print("✅ RAG和知识库组件初始化成功")
except Exception as e:
    print(f"⚠️  RAG/知识库组件初始化失败：{str(e)}")
    knowledge_base = None
    rag_engine = None
    llm = None

print("✅ 所有组件初始化完成，接口已启动（按Ctrl+C停止）")

# ========== 核心推理函数 ==========
def infer_similarity(text1: str, text2: str) -> Dict[str, Any]:
    text1, text2 = text1.strip(), text2.strip()

    # BERT推理
    with torch.no_grad():
        encoding1 = tokenizer(text1, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        encoding2 = tokenizer(text2, truncation=True, padding="max_length",
                              max_length=MAX_LEN, return_tensors="pt")
        output = model(encoding1["input_ids"], encoding1["attention_mask"],
                       encoding2["input_ids"], encoding2["attention_mask"])
        bert_sim = round(torch.clip(output, 0.0, 1.0).cpu().item(), 4)

    # 多算法相似度
    try:
        multi_sim = similarity_calculator.print_similarity_comparison(text1, text2, "", "", model)
    except:
        multi_sim = {"fused_similarity": bert_sim}

    # RAG建议
    if rag_engine is not None and llm is not None:
        try:
            rag_suggest = rag_engine.generate_suggestion(text1, llm=llm)
            suggest_source = "RAG检索增强（案例库+LoRA微调LLM）"
        except:
            rag_suggest = ""
            suggest_source = "RAG生成失败"
    else:
        rag_suggest = ""
        suggest_source = "知识库组件未初始化"

    return {
        "bert_similarity": bert_sim,
        "multi_algorithm_similarity": multi_sim,
        "repair_suggestion": rag_suggest,
        "suggest_source": suggest_source,
        "is_similar": bert_sim >= 0.5
    }

# ========== 原生HTTP请求处理 ==========
class RequestHandler(BaseHTTPRequestHandler):
    def _parse_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        except:
            return {"error": "无效的JSON格式"}

    def _send_json_response(self, status_code: int, data: Dict[str, Any]):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self._send_json_response(200, {
                "status": "success",
                "message": "工业故障匹配接口运行正常（bert-base-chinese）",
                "device": str(DEVICE),
                "bert_model": BERT_MODEL_PATH,
                "support_paths": [
                    "/predict/single（POST：单条故障匹配）",
                    "/predict/batch（POST：批量故障匹配）",
                    "/rag/suggestion（POST：RAG维修建议）"
                ]
            })
        else:
            self._send_json_response(404, {"status": "error", "message": "路径不存在"})

    def do_POST(self):
        if self.path == "/predict/single":
            data = self._parse_json()
            if "error" in data:
                self._send_json_response(400, {"status": "error", "message": data["error"]})
                return
            text1 = data.get("text1", "")
            text2 = data.get("text2", "")
            if not text1 or not text2:
                self._send_json_response(400, {"status": "error", "message": "text1和text2不能为空"})
                return
            try:
                result = infer_similarity(text1, text2)
                self._send_json_response(200, {
                    "status": "success",
                    "data": {"text1": text1, "text2": text2, **result}
                })
            except Exception as e:
                self._send_json_response(500, {"status": "error", "message": f"处理失败：{str(e)}"})

        elif self.path == "/predict/batch":
            data = self._parse_json()
            if "error" in data:
                self._send_json_response(400, {"status": "error", "message": data["error"]})
                return
            pairs = data.get("pairs", [])
            if not isinstance(pairs, list) or len(pairs) == 0:
                self._send_json_response(400, {"status": "error", "message": "pairs必须是非空列表"})
                return
            results = []
            for idx, pair in enumerate(pairs[:50]):
                text1 = pair.get("text1", "")
                text2 = pair.get("text2", "")
                if not text1 or not text2:
                    results.append({"index": idx, "error": "text1和text2不能为空"})
                    continue
                try:
                    res = infer_similarity(text1, text2)
                    results.append({"index": idx, "text1": text1, "text2": text2, **res})
                except Exception as e:
                    results.append({"index": idx, "text1": text1, "text2": text2, "error": str(e)})
            self._send_json_response(200, {
                "status": "success",
                "batch_count": len(results),
                "data": results
            })

        elif self.path == "/rag/suggestion":
            if rag_engine is None:
                self._send_json_response(500, {"status": "error", "message": "RAG引擎未初始化"})
                return
            data = self._parse_json()
            if "error" in data:
                self._send_json_response(400, {"status": "error", "message": data["error"]})
                return
            query_text = data.get("query_text", "")
            if not query_text:
                self._send_json_response(400, {"status": "error", "message": "query_text不能为空"})
                return
            try:
                similar_cases, similar_solutions = rag_engine.retrieve_similar_faults(query_text, top_k=3)
                rag_suggest = rag_engine.generate_suggestion(query_text, llm=llm)
                self._send_json_response(200, {
                    "status": "success",
                    "data": {
                        "query": query_text,
                        "similar_cases": similar_cases,
                        "rag_suggestion": rag_suggest
                    }
                })
            except Exception as e:
                self._send_json_response(500, {"status": "error", "message": f"生成失败：{str(e)}"})
        else:
            self._send_json_response(404, {"status": "error", "message": "路径不存在"})

# ========== 启动服务器 ==========
if __name__ == "__main__":
    print(f"\n🚀 启动服务：http://{APP_HOST}:{APP_PORT}")
    print("   POST /predict/single - 单条故障匹配")
    print("   POST /predict/batch  - 批量故障匹配")
    print("   POST /rag/suggestion - RAG维修建议")
    print("   GET  /             - 健康检查\n")

    server = HTTPServer((APP_HOST, APP_PORT), RequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🔌 接口已停止")
        server.server_close()