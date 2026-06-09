import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# 复用原有知识库/建议生成依赖
from knowledge_base import FaultKnowledgeBase
import warnings
warnings.filterwarnings('ignore')
# 缓存装饰器+LoRA相关依赖
from functools import lru_cache
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# 【优化】向量模型缓存（避免重复加载，提升启动速度）
@lru_cache(maxsize=1)
def _load_vector_model(model_name):
    return SentenceTransformer(model_name)

class IndustrialFaultRAGEngine:
    """工业故障场景专用RAG引擎（集成LoRA微调模型+检索增强生成）"""
    def __init__(self, knowledge_base: FaultKnowledgeBase, lora_adapter_path: str = "./lora_finetuned_qwen"):
        # 1. 初始化向量模型（不变，复用原Sentence-BERT）
        self.vector_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.vector_dim = 384  # 模型输出向量维度不变

        # 【核心修改1】初始化Chroma客户端+向量集合（替代FAISS的self.index）
        # 持久化存储向量库（数据存在本地，重启服务不丢失）
        self.chroma_client = chromadb.PersistentClient(path="./chroma_fault_vector_db")
        # 配置Chroma的嵌入函数（和原项目向量模型一致，无需手动encode）
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.vector_model_name)
        # 创建/获取向量集合（类似FAISS的Index，存储故障案例向量）
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="industrial_fault_cases",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "l2"}  # 保持和原FAISS一致的L2距离计算
        )

        # 2. 关联故障知识库（不变）
        self.kb = knowledge_base
        # 3. 加载知识库数据（不变）
        self.fault_texts, self.fault_solutions = self._load_kb_data()
        # 4. 构建向量索引（后续修改该函数，替换为Chroma逻辑）
        self._build_vector_index()

        # 5. Prompt模板（不变）
        self.prompt_template = PromptTemplate(...)

        # 6. 加载LoRA模型（不变）
        self.lora_model, self.lora_tokenizer = self._load_lora_finetuned_model(lora_adapter_path)
    def _load_kb_data(self):
        """加载知识库中的故障文本和维修方案（兜底兼容）"""
        fault_texts = []
        fault_solutions = []
        try:
            all_faults = self.kb.get_all_faults()
            for fault in all_faults:
                fault_texts.append(fault["text"].strip())
                fault_solutions.append(fault["solution"].strip())
        except:
            # 兜底：从预处理数据读取
            import pandas as pd
            train_data = pd.read_csv(r"D:\industrial_nlp_project\data\processed\train_processed.csv")
            fault_texts = train_data["fault_text1_clean"].fillna("").str.strip().tolist()
            fault_solutions = ["1. 检查设备相关部件连接状态；2. 排查故障部位是否有磨损/渗漏；3. 重启设备测试，若仍异常联系专业维修人员", ] * len(fault_texts)
        # 过滤空文本
        valid_pairs = [(t, s) for t, s in zip(fault_texts, fault_solutions) if t]
        fault_texts = [t for t, s in valid_pairs]
        fault_solutions = [s for t, s in valid_pairs]
        return fault_texts, fault_solutions

    def _build_vector_index(self):
        """构建Chroma向量索引（替代原FAISS逻辑）"""
        if len(self.fault_texts) == 0:
            raise Exception("知识库无有效故障数据，无法构建RAG向量库")
        
        # 【核心修改2】向Chroma添加故障案例（自动完成向量编码，无需手动encode）
        self.chroma_collection.add(
            documents=self.fault_texts,  # 故障文本（Chroma会自动用embedding_fn编码）
            metadatas=[{"solution": sol} for sol in self.fault_solutions],  # 关联维修方案
            ids=[f"fault_case_{i}" for i in range(len(self.fault_texts))]  # 唯一ID（用于后续关联）
        )
        
        print(f"✅ Chroma向量库构建完成：共{self.chroma_collection.count()}条故障案例，向量维度{self.vector_dim}")

    def _load_lora_finetuned_model(self, lora_adapter_path):
        """加载LoRA适配器+基础模型（修复函数嵌套错误）"""
        try:
            # 加载LoRA配置
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)
            # 4bit量化配置
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            # 加载LoRA适配器
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            # 加载Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=True,
                bos_token="<s>",
                eos_token="</s>",
                pad_token="<pad>"
            )
            print(f"✅ 成功加载LoRA微调模型，适配器路径：{lora_adapter_path}")
            return model, tokenizer
        except Exception as e:
            print(f"⚠️ LoRA模型加载失败，降级为LangChain LLM调用：{str(e)}")
            return None, None

    def retrieve_similar_faults(self, query_text, top_k=3, threshold=0.5):
        """检索相似故障案例（Chroma实现，替代原FAISS）"""
        query_text = query_text.strip()
        if not query_text:
            return [], []
        
        # 【核心修改3】Chroma检索（自动编码查询文本，返回相似结果）
        results = self.chroma_collection.query(
            query_texts=[query_text],  # 用户查询文本
            n_results=top_k,  # 取Top-K相似案例
            include=["documents", "metadatas", "distances"]  # 返回文本、维修方案、距离
        )

        # 解析检索结果（映射为原函数返回格式，不影响后续生成逻辑）
        similar_cases = results["documents"][0]  # 相似故障文本
        similar_solutions = [meta["solution"] for meta in results["metadatas"][0]]  # 对应维修方案
        distances = results["distances"][0]  # L2距离（和原FAISS一致）

        # 过滤低相似度案例（和原逻辑一致：距离越小越相似，转换为相似度过滤）
        valid_pairs = []
        for case, sol, dist in zip(similar_cases, similar_solutions, distances):
            similarity = 1 - (dist / np.sqrt(self.vector_dim))  # L2距离转相似度（0-1）
            if similarity >= threshold:
                valid_pairs.append((case, sol))
        
        return [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]

    def generate_suggestion(self, query_text, llm=None):
        """RAG增强生成维修建议（优先LoRA模型，多级降级）"""
        # 1. 检索相似案例
        similar_cases, similar_solutions = self.retrieve_similar_faults(query_text)
        # 2. 构建Prompt
        prompt = self.prompt_template.format(
            query=query_text,
            similar_cases="\n".join([f"{i+1}. {case}" for i, case in enumerate(similar_cases)]) or "无",
            similar_solutions="\n".join([f"{i+1}. {sol}" for i, sol in enumerate(similar_solutions)]) or "无"
        )
        # 3. 优先使用LoRA微调模型
        if self.lora_model is not None and self.lora_tokenizer is not None:
            inputs = self.lora_tokenizer(
                prompt,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            # 生成配置（平衡精准度和随机性）
            outputs = self.lora_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.6,
                do_sample=True,
                eos_token_id=self.lora_tokenizer.eos_token_id,
                pad_token_id=self.lora_tokenizer.pad_token_id
            )
            # 解码清理
            suggestion = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "维修建议：" in suggestion:
                suggestion = suggestion.split("维修建议：")[-1].strip()
            return suggestion
        # 4. 降级：使用传入的LangChain LLM
        elif llm is not None:
            chain = LLMChain(llm=llm, prompt=self.prompt_template)
            suggestion = chain.run(
                query=query_text,
                similar_cases="\n".join(similar_cases) or "无",
                similar_solutions="\n".join(similar_solutions) or "无"
            )
            return suggestion.strip()
        # 5. 最终兜底：直接返回相似案例建议
        else:
            if similar_solutions:
                base_suggestion = f"参考相似故障处理方案：\n{chr(10).join(similar_solutions[:2])}\n\n注意：按步骤操作，避免违规拆卸设备"
            else:
                base_suggestion = "1. 检查设备电源/连接是否正常；2. 观察故障部位有无物理损伤、渗漏或异响；3. 记录故障发生时间和场景，联系专业维修人员进行深度排查"
            return base_suggestion
    def _build_vector_index(self):
        """构建FAISS向量索引"""
        if not self.fault_texts:
            warnings.warn("无故障文本，向量库构建失败")
            return
        # 生成向量
        vectors = self.vector_model.encode(self.fault_texts, normalize_embeddings=True)
        # 添加到索引
        self.index.add(np.array(vectors).astype(np.float32))
        print(f"✅ 向量库构建完成，包含{self.index.ntotal}条故障文本向量")

    def retrieve_similar_cases(self, query, top_k=3, threshold=0.5):
        """检索相似故障案例"""
        # 生成查询向量
        query_vector = self.vector_model.encode([query], normalize_embeddings=True)
        # 检索
        distances, indices = self.index.search(np.array(query_vector).astype(np.float32), top_k)
        # 过滤低相似度（L2距离越小越相似，转换为相似度）
        similar_cases = []
        similar_solutions = []
        for i, idx in enumerate(indices[0]):
            if idx < 0: continue
            similarity = 1 - (distances[0][i] / np.sqrt(self.vector_dim))  # L2转相似度
            if similarity >= threshold:
                similar_cases.append(self.fault_texts[idx])
                similar_solutions.append(self.fault_solutions[idx])
        return similar_cases, similar_solutions

    def generate_suggestion(self, query):
        """检索+生成维修建议"""
        # 1. 检索相似案例
        similar_cases, similar_solutions = self.retrieve_similar_cases(query)
        # 2. 格式化Prompt
        prompt = self.prompt_template.format(
            query=query,
            similar_cases="；".join(similar_cases) if similar_cases else "无",
            similar_solutions="；".join(similar_solutions) if similar_solutions else "无"
        )
        # 3. LLM生成
        inputs = self.lora_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        outputs = self.lora_model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        response = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

