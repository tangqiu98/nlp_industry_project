---
name: prepare_bert_siamese_data
description: |
  将工业故障原始数据转换为BERT双塔(Siamese)模型训练所需的正负样本对和标签。
  支持相似度阈值划分、难例挖掘、数据增强等功能，输出可直接用于bert.py训练的CSV格式。
type: data_processing
version: "1.0.0"

args:
  - name: raw_data_path
    type: string
    required: true
    description: 原始数据CSV文件路径，需包含fault_text1, fault_text2, semantic_similarity列

  - name: output_dir
    type: string
    required: true
    description: 处理后数据的输出目录

  - name: positive_threshold
    type: float
    default: 0.7
    description: 正样本阈值，相似度>=此值视为正样本对

  - name: negative_threshold
    type: float
    default: 0.3
    description: 负样本阈值，相似度<=此值视为负样本对

  - name: max_seq_length
    type: int
    default: 64
    description: BERT最大序列长度

  - name: hard_negative_ratio
    type: float
    default: 0.2
    description: 难负例比例(0.3-0.7相似度区间)，用于提升模型区分能力

  - name: augment_positive
    type: boolean
    default: false
    description: 是否对正样本进行同义词替换数据增强

  - name: train_ratio
    type: float
    default: 0.8
    description: 训练集比例

  - name: val_ratio
    type: float
    default: 0.1
    description: 验证集比例

examples:
  - description: 基础用法，使用默认参数处理数据
    args:
      raw_data_path: "data/raw/fault_semantic_matching_train.csv"
      output_dir: "data/processed"

  - description: 自定义阈值并启用难例挖掘
    args:
      raw_data_path: "data/raw/fault_semantic_matching_train.csv"
      output_dir: "data/processed"
      positive_threshold: 0.75
      negative_threshold: 0.35
      hard_negative_ratio: 0.3
      augment_positive: true

output:
  format: csv
  files:
    - train_processed.csv
    - validation_processed.csv
    - test_processed.csv
  columns:
    - fault_text1_clean: 清洗后的故障文本1
    - fault_text2_clean: 清洗后的故障文本2
    - label: 二分类标签(1=相似, 0=不相似)
    - similarity_score: 原始相似度分数(可选)
      """

import os
import re
import csv
import random
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SamplePair:
    """样本对数据类"""
    text1: str
    text2: str
    label: int
    similarity: float
    pair_type: str  # 'positive', 'negative', 'hard_negative'


@dataclass
class ProcessingConfig:
    """处理配置"""
    positive_threshold: float = 0.7
    negative_threshold: float = 0.3
    max_seq_length: int = 64
    hard_negative_ratio: float = 0.2
    augment_positive: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    random_seed: int = 42


class IndustrialTextCleaner:
    """工业故障文本清洗器"""

    # 工业领域术语归一化词典
    INDUSTRIAL_TERMS = {
        # 设备类
        r'\b油缸\b': '液压缸',
        r'\b油压缸\b': '液压缸',
        r'\b液压筒\b': '液压缸',
        r'\b缓冲器\b': '缓冲装置',
        r'\b减震器\b': '缓冲装置',
        # 故障类
        r'\b渗漏\b': '泄漏',
        r'\b滴漏\b': '泄漏',
        r'\b异音\b': '异响',
        r'\b卡滞\b': '卡死',
        r'\b抱死\b': '卡死',
        r'\b磨耗\b': '磨损',
        # 程度类
        r'\b失效\b': '故障',
        r'\b失灵\b': '故障',
        r'\b损坏\b': '故障',
    }

    # 停用词列表
    STOPWORDS = {
        '的', '了', '是', '在', '有', '和', '就', '都', '而', '及', '与', '着', '过',
        '要', '不', '没', '也', '很', '一个', '可以', '可能', '这个', '那个', '什么',
        '怎么', '为什么', '如何', '哪', '哪个', '哪些'
    }

    def __init__(self):
        self.compiled_terms = {
            pattern: replacement
            for pattern, replacement in self.INDUSTRIAL_TERMS.items()
        }

    def clean(self, text: str) -> str:
        """清洗文本"""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.strip()

        # 术语归一化
        for pattern, replacement in self.compiled_terms.items():
            text = re.sub(pattern, replacement, text)

        # 去除特殊字符，保留中文、字母、数字
        text = re.sub(r'[^一-龥a-zA-Z0-9\s]', ' ', text)

        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_stopwords(self, text: str) -> List[str]:
        """去除停用词"""
        words = text.split()
        return [w for w in words if w not in self.STOPWORDS]


class DataAugmenter:
    """数据增强器"""

    # 同义词词典
    SYNONYMS = {
        '故障': ['问题', '异常', '失效', '损坏'],
        '泄漏': ['渗漏', '滴漏', '漏油', '漏气'],
        '异响': ['异音', '噪音', '噪声', '振动声'],
        '卡死': ['卡滞', '抱死', '卡住', '停滞'],
        '磨损': ['磨耗', '损耗', '磨蚀', '损坏'],
        '温度高': ['温升过高', '过热', '高温', '温度异常'],
        '无输出': ['无信号', ['输出', '失效'], '输出异常'],
        '通讯故障': ['通信异常', '连接中断', '通讯中断'],
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """同义词替换增强"""
        words = list(text)
        replaceable = []

        for term, synonyms in self.SYNONYMS.items():
            if term in text:
                replaceable.append((term, synonyms))

        if not replaceable:
            return text

        # 随机选择n个词进行替换
        n = min(n, len(replaceable))
        selected = random.sample(replaceable, n)

        for term, synonyms in selected:
            synonym = random.choice(synonyms)
            text = text.replace(term, synonym)

        return text

    def random_swap(self, text: str) -> str:
        """随机交换相邻词语"""
        words = text.split()
        if len(words) <= 2:
            return text

        i = random.randint(0, len(words) - 2)
        words[i], words[i + 1] = words[i + 1], words[i]
        return ''.join(words)

    def augment(self, text: str, num_augments: int = 2) -> List[str]:
        """生成多个增强样本"""
        augmented = []

        # 同义词替换
        aug1 = self.synonym_replacement(text, n=1)
        if aug1 != text:
            augmented.append(aug1)

        # 随机交换
        aug2 = self.random_swap(text)
        if aug2 != text:
            augmented.append(aug2)

        return augmented[:num_augments]


class BertSiameseDataPreparer:
    """BERT双塔模型数据准备器"""

    def __init__(
        self,
        raw_data_path: str,
        output_dir: str,
        config: Optional[ProcessingConfig] = None
    ):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.config = config or ProcessingConfig()

        self.cleaner = IndustrialTextCleaner()
        self.augmenter = DataAugmenter(seed=self.config.random_seed)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 统计数据
        self.stats = {
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'hard_negative_samples': 0,
            'augmented_samples': 0,
        }

    def load_raw_data(self) -> List[Dict]:
        """加载原始数据"""
        logger.info(f"正在加载原始数据: {self.raw_data_path}")

        samples = []
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']

        for encoding in encodings:
            try:
                with open(self.raw_data_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        samples.append(row)
                logger.info(f"成功使用 {encoding} 编码加载 {len(samples)} 条数据")
                break
            except UnicodeDecodeError:
                continue

        if not samples:
            raise ValueError(f"无法读取数据文件: {self.raw_data_path}")

        return samples

    def create_sample_pairs(self, data: List[Dict]) -> List[SamplePair]:
        """创建样本对"""
        logger.info("正在创建样本对...")

        pairs = []
        fault_dict = defaultdict(list)  # 用于难例挖掘

        for item in data:
            try:
                text1 = self.cleaner.clean(item.get('fault_text1', ''))
                text2 = self.cleaner.clean(item.get('fault_text2', ''))
                similarity = float(item.get('semantic_similarity', 0))

                if not text1 or not text2:
                    continue

                # 根据阈值划分正负样本
                if similarity >= self.config.positive_threshold:
                    label = 1
                    pair_type = 'positive'
                    self.stats['positive_samples'] += 1

                    # 正样本增强
                    if self.config.augment_positive:
                        aug_texts = self.augmenter.augment(text1)
                        for aug_text in aug_texts:
                            pairs.append(SamplePair(
                                text1=aug_text,
                                text2=text2,
                                label=1,
                                similarity=similarity,
                                pair_type='augmented_positive'
                            ))
                            self.stats['augmented_samples'] += 1

                elif similarity <= self.config.negative_threshold:
                    label = 0
                    pair_type = 'negative'
                    self.stats['negative_samples'] += 1

                else:
                    # 难负例 (相似度在中间区间)
                    if random.random() < self.config.hard_negative_ratio:
                        label = 0
                        pair_type = 'hard_negative'
                        self.stats['hard_negative_samples'] += 1
                    else:
                        continue  # 丢弃中间区间样本

                pairs.append(SamplePair(
                    text1=text1,
                    text2=text2,
                    label=label,
                    similarity=similarity,
                    pair_type=pair_type
                ))

                # 记录用于后续难例挖掘
                fault_dict[text1[:10]].append((text2, similarity))

                self.stats['total_samples'] += 1

            except (ValueError, KeyError) as e:
                logger.warning(f"跳过无效数据行: {e}")
                continue

        logger.info(f"样本对创建完成: 共 {len(pairs)} 对")
        logger.info(f"  - 正样本: {self.stats['positive_samples']}")
        logger.info(f"  - 负样本: {self.stats['negative_samples']}")
        logger.info(f"  - 难负例: {self.stats['hard_negative_samples']}")
        logger.info(f"  - 增强样本: {self.stats['augmented_samples']}")

        return pairs

    def create_hard_negatives(self, pairs: List[SamplePair]) -> List[SamplePair]:
        """创建难负例 - 选择相似度略低于正样本阈值的样本对"""
        logger.info("正在挖掘难负例...")

        # 按相似度排序，选择0.5-0.7区间的作为难负例
        hard_negatives = []
        for pair in pairs:
            if 0.5 <= pair.similarity < self.config.positive_threshold:
                pair.label = 0
                pair.pair_type = 'hard_negative'
                hard_negatives.append(pair)

        # 限制难负例数量
        max_hard = int(len(pairs) * self.config.hard_negative_ratio)
        hard_negatives = hard_negatives[:max_hard]

        logger.info(f"挖掘到 {len(hard_negatives)} 个难负例")
        return hard_negatives

    def split_dataset(
        self,
        pairs: List[SamplePair]
    ) -> Tuple[List[SamplePair], List[SamplePair], List[SamplePair]]:
        """划分训练集、验证集、测试集（保持标签分布）"""
        logger.info("正在划分数据集...")

        # 按pair_type分组
        positive_pairs = [p for p in pairs if p.label == 1]
        negative_pairs = [p for p in pairs if p.label == 0]

        # 打乱
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)

        # 计算分割点
        train_pos = int(len(positive_pairs) * self.config.train_ratio)
        val_pos = int(len(positive_pairs) * (self.config.train_ratio + self.config.val_ratio))

        train_neg = int(len(negative_pairs) * self.config.train_ratio)
        val_neg = int(len(negative_pairs) * (self.config.train_ratio + self.config.val_ratio))

        # 分割
        train_pairs = positive_pairs[:train_pos] + negative_pairs[:train_neg]
        val_pairs = positive_pairs[train_pos:val_pos] + negative_pairs[train_neg:val_neg]
        test_pairs = positive_pairs[val_pos:] + negative_pairs[val_neg:]

        # 打乱各数据集
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
        random.shuffle(test_pairs)

        logger.info(f"数据集划分完成:")
        logger.info(f"  - 训练集: {len(train_pairs)} 对 (正:{sum(1 for p in train_pairs if p.label==1)}, 负:{sum(1 for p in train_pairs if p.label==0)})")
        logger.info(f"  - 验证集: {len(val_pairs)} 对 (正:{sum(1 for p in val_pairs if p.label==1)}, 负:{sum(1 for p in val_pairs if p.label==0)})")
        logger.info(f"  - 测试集: {len(test_pairs)} 对 (正:{sum(1 for p in test_pairs if p.label==1)}, 负:{sum(1 for p in test_pairs if p.label==0)})")

        return train_pairs, val_pairs, test_pairs

    def save_to_csv(
        self,
        pairs: List[SamplePair],
        filename: str
    ) -> str:
        """保存为CSV文件"""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fault_text1_clean',
                'fault_text2_clean',
                'label',
                'similarity_score',
                'pair_type'
            ])

            for pair in pairs:
                writer.writerow([
                    pair.text1,
                    pair.text2,
                    pair.label,
                    round(pair.similarity, 4),
                    pair.pair_type
                ])

        logger.info(f"已保存: {filepath} ({len(pairs)} 条)")
        return filepath

    def save_stats(self) -> str:
        """保存统计信息"""
        filepath = os.path.join(self.output_dir, 'processing_stats.json')

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        logger.info(f"已保存统计信息: {filepath}")
        return filepath

    def prepare(self) -> Dict[str, str]:
        """执行完整的数据准备流程"""
        logger.info("=" * 50)
        logger.info("开始BERT双塔模型数据准备")
        logger.info("=" * 50)

        # 1. 加载原始数据
        raw_data = self.load_raw_data()

        # 2. 创建样本对
        pairs = self.create_sample_pairs(raw_data)

        # 3. 挖掘难负例
        hard_negatives = self.create_hard_negatives(raw_data)
        all_pairs = pairs + hard_negatives

        # 4. 划分数据集
        train_pairs, val_pairs, test_pairs = self.split_dataset(all_pairs)

        # 5. 保存
        train_path = self.save_to_csv(train_pairs, 'train_processed.csv')
        val_path = self.save_to_csv(val_pairs, 'validation_processed.csv')
        test_path = self.save_to_csv(test_pairs, 'test_processed.csv')
        stats_path = self.save_stats()

        logger.info("=" * 50)
        logger.info("数据准备完成!")
        logger.info("=" * 50)

        return {
            'train_path': train_path,
            'val_path': val_path,
            'test_path': test_path,
            'stats_path': stats_path
        }


def main():
    """主入口函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BERT双塔模型数据准备工具')
    parser.add_argument('--raw_data_path', type=str, required=True, help='原始数据路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--positive_threshold', type=float, default=0.7, help='正样本阈值')
    parser.add_argument('--negative_threshold', type=float, default=0.3, help='负样本阈值')
    parser.add_argument('--hard_negative_ratio', type=float, default=0.2, help='难负例比例')
    parser.add_argument('--augment_positive', action='store_true', help='启用数据增强')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # 创建配置
    config = ProcessingConfig(
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
        hard_negative_ratio=args.hard_negative_ratio,
        augment_positive=args.augment_positive,
        random_seed=args.seed
    )

    # 执行数据准备
    preparer = BertSiameseDataPreparer(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        config=config
    )

    results = preparer.prepare()

    print("\n" + "=" * 50)
    print("处理完成! 输出文件:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    print("=" * 50)


if __name__ == '__main__':
    main()
