"""
RAG召回评估脚本
评估RAG系统的检索召回效果

数据来源: fault_semantic_matching_test.csv (测试集)
评估指标: Retrieval Recall, NDCG, MAP

使用原生TF-IDF实现（无需下载外部模型）
"""

import os
import sys
import re
import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 路径配置
PROJECT_PATH = r"d:\桌面\industrial_nlp_project (2)\industrial_nlp_project"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "data")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============== 原生TF-IDF实现 ==============

class NativeTfidfVectorizer:
    """原生TF-IDF向量器"""
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else set()
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0

    def _tokenize(self, text):
        text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', text)
        tokens = [token for token in text.split() if token and token not in self.stop_words]
        return tokens

    def fit(self, documents):
        self.doc_count = len(documents)
        doc_freq = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
        self.vocab = {token: idx for idx, token in enumerate(doc_freq.keys())}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1

    def transform(self, documents):
        tfidf_vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            if not tokens:
                tfidf_vectors.append({})
                continue
            tf = Counter(tokens)
            total_tokens = len(tokens)
            vector = {}
            for token, count in tf.items():
                if token in self.vocab:
                    tf_val = count / total_tokens
                    idf_val = self.idf.get(token, 1.0)
                    vector[self.vocab[token]] = tf_val * idf_val
            tfidf_vectors.append(vector)
        return tfidf_vectors

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

def cosine_similarity(vec1, vec2):
    """计算两个稀疏向量的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    dot_product = 0.0
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
    for idx, val in vec1.items():
        if idx in vec2:
            dot_product += val * vec2[idx]
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# ============== 评估指标函数 ==============

def calculate_retrieval_recall(query, relevant_docs, retrieved_docs):
    """
    计算检索召回率：检索到的相关文档数 / 所有相关文档数
    """
    relevant_in_retrieved = len(set(relevant_docs) & set(retrieved_docs))
    total_relevant = len(relevant_docs)
    return relevant_in_retrieved / total_relevant if total_relevant > 0 else 0.0

def calculate_ndcg(query, relevant_docs, retrieved_docs, relevance_scores=None):
    """
    计算NDCG：考虑检索结果排序质量的指标
    """
    if relevance_scores is None:
        relevance_scores = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]

    # DCG@k
    dcg = sum(rel / np.log2(i+2) for i, rel in enumerate(relevance_scores))

    # IDCG@k（理想情况下的最大DCG）
    ideal_scores = sorted([1]*len(relevant_docs) + [0]*(len(retrieved_docs)-len(relevant_docs)), reverse=True)
    idcg = sum(rel / np.log2(i+2) for i, rel in enumerate(ideal_scores))

    return dcg / idcg if idcg > 0 else 0.0

def calculate_map(query_results):
    """
    计算MAP：平均精确率的均值
    """
    aps = []
    for query, relevant_docs, retrieved_docs in query_results:
        precision_scores = []
        relevant_found = 0
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                relevant_found += 1
                precision_scores.append(relevant_found / (i+1))
        ap = sum(precision_scores) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        aps.append(ap)
    return sum(aps) / len(aps) if len(aps) > 0 else 0.0

# ============== RAG检索器 (使用TF-IDF) ==============

class SimpleRAGRetriever:
    """简化的RAG检索器（基于TF-IDF）"""

    def __init__(self, documents):
        self.documents = documents

        # 中文停用词
        stop_words = ["的", "了", "是", "在", "有", "和", "就", "都", "而", "及", "与", "着", "过", "要", "不", "没", "也", "很", "为", "之", "以", "于", "但", "或", "等", "让", "被", "把", "给", "让"]

        # 初始化TF-IDF
        self.tfidf = NativeTfidfVectorizer(stop_words=stop_words)

        # 预计算所有文档向量
        print("正在编码文档向量...")
        self.doc_vectors = self.tfidf.fit_transform(documents)
        print(f"已编码 {len(documents)} 个文档")

    def retrieve(self, query, top_k=5):
        """检索与query最相似的top_k个文档"""
        query_vector = self.tfidf.transform([query])[0]
        similarities = []
        for doc_vec in self.doc_vectors:
            sim = cosine_similarity(query_vector, doc_vec)
            similarities.append(sim)
        # 获取top_k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_docs = [self.documents[i] for i in top_indices]
        return retrieved_docs

# ============== 主评估流程 ==============

def run_recall_evaluation():
    """运行RAG召回评估"""

    # 加载测试数据
    test_file = os.path.join(DATA_PATH, "fault_semantic_matching_test.csv")
    df_test = pd.read_csv(test_file)
    print(f"加载测试数据: {len(df_test)} 条")

    # 加载知识库文档 (fault_text_preprocessed)
    kb_file = os.path.join(DATA_PATH, "fault_text_preprocessed.csv")
    df_kb = pd.read_csv(kb_file)
    # 使用filtered_text作为知识库文档
    kb_documents = df_kb['filtered_text'].dropna().str.strip().tolist()
    kb_documents = [d for d in kb_documents if d]
    print(f"知识库文档数(原始): {len(kb_documents)}")

    # 扩展知识库：将测试集中的fault_text1也加入知识库（作为候选检索文档）
    # 这样可以评估检索系统能否从扩展知识库中召回相关文档
    all_fault_texts = list(set(df_test['fault_text1'].tolist() + df_test['fault_text2'].tolist() + kb_documents))
    print(f"扩展后知识库文档数: {len(all_fault_texts)}")

    # 初始化RAG检索器（使用扩展知识库）
    retriever = SimpleRAGRetriever(all_fault_texts)

    # 定义相关文档阈值 (相似度>0.5认为相关)
    SIMILARITY_THRESHOLD = 0.5

    # 评估结果收集
    all_recalls = []
    all_ndcgs = []
    query_results_for_map = []

    # 预构建 query -> {relevant_docs} 的映射（用于评估）
    # 对于每个fault_text1，找到所有与之语义相似度>=threshold的fault_text2
    query_to_relevant = {}
    for idx, row in df_test.iterrows():
        q = row['fault_text1'].strip()
        candidate = row['fault_text2'].strip()
        sim = row['semantic_similarity']
        if sim >= SIMILARITY_THRESHOLD:
            if q not in query_to_relevant:
                query_to_relevant[q] = set()
            query_to_relevant[q].add(candidate)

    print(f"有相关文档的查询数: {len(query_to_relevant)}")

    # 对测试集中的每对故障文本进行评估
    print("\n开始评估...")
    print("-" * 60)

    evaluated_count = 0
    for idx, row in df_test.iterrows():
        query = row['fault_text1'].strip()
        candidate_doc = row['fault_text2'].strip()
        similarity = row['semantic_similarity']

        # 只评估有明确相关性的样本
        if similarity < SIMILARITY_THRESHOLD:
            continue

        # 执行检索 - 从扩展知识库中检索与query相似的文档
        retrieved_docs = retriever.retrieve(query, top_k=5)

        # 相关文档：从测试集中找到的所有与query相似的fault_text2
        relevant_docs = list(query_to_relevant.get(query, set()))

        # 计算Recall: 检索到的相关文档数 / 所有相关文档数
        recall = calculate_retrieval_recall(query, relevant_docs, retrieved_docs)
        all_recalls.append(recall)

        # 计算NDCG: 使用similarity作为相关性分数
        relevance_scores = [similarity if doc in relevant_docs else 0 for doc in retrieved_docs]
        ndcg = calculate_ndcg(query, relevant_docs, retrieved_docs, relevance_scores)
        all_ndcgs.append(ndcg)

        # 收集MAP数据
        query_results_for_map.append((query, relevant_docs, retrieved_docs))

        evaluated_count += 1
        if evaluated_count % 50 == 0:
            print(f"已评估: {evaluated_count} 条")

    print(f"实际评估样本数: {evaluated_count}")

    # 计算MAP
    map_score = calculate_map(query_results_for_map)

    # 汇总结果
    avg_recall = np.mean(all_recalls) if all_recalls else 0.0
    avg_ndcg = np.mean(all_ndcgs) if all_ndcgs else 0.0

    # ============== 打印并保存结果 ==============

    results = []
    results.append("=" * 70)
    results.append("RAG 召回评估报告")
    results.append("=" * 70)
    results.append(f"\n评估数据: fault_semantic_matching_test.csv")
    results.append(f"知识库文档数: {len(kb_documents)}")
    results.append(f"有效评估样本数: {evaluated_count}")
    results.append(f"检索Top-K: 5")
    results.append(f"相关文档阈值: {SIMILARITY_THRESHOLD}")
    results.append("\n" + "-" * 70)
    results.append("评估指标结果")
    results.append("-" * 70)
    results.append(f"1. Retrieval Recall (检索召回率): {avg_recall:.4f}")
    results.append(f"2. NDCG (归一化折损累计增益): {avg_ndcg:.4f}")
    results.append(f"3. MAP (平均精确率均值): {map_score:.4f}")
    results.append("\n" + "-" * 70)
    results.append("评估指标说明")
    results.append("-" * 70)
    results.append("- Retrieval Recall: 检索到的相关文档数 / 所有相关文档数")
    results.append("- NDCG: 考虑检索结果排序质量的指标，越高越好")
    results.append("- MAP: 平均精确率的均值，衡量检索系统整体准确性")
    results.append("\n" + "-" * 70)
    results.append("详细评估数据 (前20条)")
    results.append("-" * 70)

    # 打印详细结果 (前20条)
    detail_header = f"{'Query':<30} | {'Relevant':<20} | {'Retrieved Top-1':<25} | Recall"
    results.append(detail_header)
    results.append("-" * 70)

    for i, (q, rel, ret) in enumerate(query_results_for_map[:20]):
        query_short = q[:27] + "..." if len(q) > 30 else q
        rel_short = rel[0][:17] + "..." if rel and len(rel[0]) > 20 else (rel[0] if rel else "N/A")
        ret_short = ret[0][:22] + "..." if len(ret[0]) > 25 else ret[0]
        recall_val = calculate_retrieval_recall(q, rel, ret)
        results.append(f"{query_short:<30} | {rel_short:<20} | {ret_short:<25} | {recall_val:.2f}")

    results.append("\n" + "=" * 70)
    results.append("评估完成")
    results.append("=" * 70)

    # 打印结果
    output_text = "\n".join(results)
    print(output_text)

    # 保存到文件
    output_file = os.path.join(OUTPUT_PATH, "召回评估.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)

    print(f"\n结果已保存到: {output_file}")

    return {
        'recall': avg_recall,
        'ndcg': avg_ndcg,
        'map': map_score,
        'sample_count': evaluated_count
    }

if __name__ == "__main__":
    results = run_recall_evaluation()