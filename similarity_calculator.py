import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

# ========== 原生TF-IDF实现（替换sklearn的TfidfVectorizer） ==========
class NativeTfidfVectorizer:
    def __init__(self, stop_words: List[str] = None):
        self.stop_words = set(stop_words) if stop_words else set()
        self.vocab: Dict[str, int] = {}  # 词→索引映射
        self.idf: Dict[str, float] = {}  # 词→idf值
        self.doc_count = 0  # 文档总数

    # 分词（简单中文分词，按字符+停用词过滤）
    def _tokenize(self, text: str) -> List[str]:
        # 过滤非中文字符，保留中文、字母、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        # 按空格分割，过滤停用词和空字符串
        tokens = [token for token in text.split() if token and token not in self.stop_words]
        return tokens

    # 拟合文档，计算IDF
    def fit(self, documents: List[str]):
        self.doc_count = len(documents)
        # 统计每个词在多少个文档中出现（文档频率）
        doc_freq = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc))  # 去重，每个文档只算一次
            for token in tokens:
                doc_freq[token] += 1
        # 构建词汇表和IDF
        self.vocab = {token: idx for idx, token in enumerate(doc_freq.keys())}
        # IDF公式：log((文档总数 + 1) / (词出现的文档数 + 1)) + 1（平滑处理）
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1

    # 转换文本为TF-IDF向量
    def transform(self, documents: List[str]) -> List[Dict[int, float]]:
        tfidf_vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            if not tokens:
                tfidf_vectors.append({})
                continue
            # 计算TF（词频）：词在文档中出现的次数 / 文档总词数
            tf = Counter(tokens)
            total_tokens = len(tokens)
            # 计算TF-IDF：tf * idf
            vector = {}
            for token, count in tf.items():
                if token in self.vocab:
                    tf_val = count / total_tokens
                    idf_val = self.idf.get(token, 1.0)  # 未见过的词IDF设为1.0
                    vector[self.vocab[token]] = tf_val * idf_val
            tfidf_vectors.append(vector)
        return tfidf_vectors

    # 拟合+转换（简化接口，与sklearn兼容）
    def fit_transform(self, documents: List[str]) -> List[Dict[int, float]]:
        self.fit(documents)
        return self.transform(documents)

# ========== 余弦相似度计算（原生实现，无需sklearn） ==========
def cosine_similarity(vec1: Dict[int, float], vec2: Dict[int, float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    # 计算分子：点积
    dot_product = 0.0
    # 遍历较短的向量以提高效率
    if len(vec1) > len(vec2):
        vec1, vec2 = vec2, vec1
    for idx, val in vec1.items():
        if idx in vec2:
            dot_product += val * vec2[idx]
    # 计算分母：两个向量的L2范数乘积
    norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val**2 for val in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(dot_product / (norm1 * norm2), 4)

# ========== 原MultiSimilarityCalculator类（仅修改TF-IDF初始化） ==========
class MultiSimilarityCalculator:
    def __init__(self):
        # 替换sklearn的TfidfVectorizer为原生实现
        self.tfidf = NativeTfidfVectorizer(
            stop_words=["的", "了", "是", "在", "有", "和", "就", "都", "而", "及", "与", "着", "过", "要", "不", "没", "也", "很"]
        )
        # 示例文档（用于拟合TF-IDF，可根据实际场景补充）
        sample_docs = [
            "液压缸泄漏，无法加压", "油缸渗漏，压力不足", "电机不转，无法启动",
            "马达无法启动，无响应", "轴承磨损，异响严重", "齿轮磨损，转动异响"
        ]
        self.tfidf.fit(sample_docs)

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """计算余弦相似度（TF-IDF）"""
        vectors = self.tfidf.transform([text1, text2])
        return cosine_similarity(vectors[0], vectors[1])

    def calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算Jaccard相似度（词集交集/并集）"""
        tokenizer = self.tfidf._tokenize
        set1 = set(tokenizer(text1))
        set2 = set(tokenizer(text2))
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return round(intersection / union, 4)

    def print_similarity_comparison(self, text1: str, text2: str, _, __, ___) -> Dict[str, float]:
        """原接口兼容，返回多算法相似度结果"""
        cos_sim = self.calculate_cosine_similarity(text1, text2)
        jaccard_sim = self.calculate_jaccard_similarity(text1, text2)
        # 加权融合（余弦相似度权重0.7，Jaccard权重0.3）
        fused_sim = round(0.7 * cos_sim + 0.3 * jaccard_sim, 4)
        return {
            "cosine_similarity": cos_sim,
            "jaccard_similarity": jaccard_sim,
            "fused_similarity": fused_sim
        }