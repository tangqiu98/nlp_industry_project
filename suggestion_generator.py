import os
import json
import warnings
import ollama  # 新增：导入ollama库（本地部署qwen:0.5b需提前安装）
warnings.filterwarnings('ignore')

# 所有路径统一为 D:\industrial_nlp_project
BASE_PATH = "D:\\industrial_nlp_project"
# 新增：ollama配置（本地部署qwen:0.5b，无需API密钥，不消耗个人算力）
OLLAMA_MODEL = "qwen:0.5b"  # 固定本地部署的qwen:0.5b模型
OLLAMA_TIMEOUT = 30  # 超时时间（避免模型响应过慢）

class RepairSuggestionGenerator:
    def __init__(self, knowledge_base):
        """
        初始化维修建议生成器
        :param knowledge_base: FaultKnowledgeBase实例（关联知识库）
        """
        self.kb = knowledge_base
        # 维修建议优先级（应急→修复→预防→备件）
        self.priority = ["应急处理", "根本修复", "预防措施", "备件清单"]
        # 新增：初始化ollama客户端（本地连接，无需额外配置）
        self.ollama_client = ollama.Client(timeout=OLLAMA_TIMEOUT)
    
    def _validate_suggest(self, suggest_text):
        """校验维修建议有效性（过滤无效建议）"""
        if not suggest_text.strip():
            return False
        # 工业维修建议关键词校验
        valid_keywords = ["更换", "紧固", "检测", "调整", "清洗", "检查", "维修", "更换", "备用"]
        return any(keyword in suggest_text for keyword in valid_keywords)
    
    def _parse_suggest(self, suggest_text):
        """解析维修建议（支持字符串/JSON格式）"""
        if not suggest_text.strip():
            return {}
        
        # 尝试JSON解析（优先）
        try:
            suggest_dict = json.loads(suggest_text)
            # 校验格式（是否包含优先级字段）
            for key in self.priority:
                if key not in suggest_dict:
                    suggest_dict[key] = ""
            return suggest_dict
        except:
            # 字符串格式，按分隔符拆分（妥协逻辑，兼容企业手动录入）
            separators = ["|", "；", "；", "\n", " "]
            suggest_list = []
            for sep in separators:
                if sep in suggest_text:
                    suggest_list = [s.strip() for s in suggest_text.split(sep) if s.strip()]
                    break
            if not suggest_list:
                suggest_list = [suggest_text.strip()]
            
            # 按优先级分配（简化，前4条对应优先级）
            suggest_dict = {}
            for i, key in enumerate(self.priority):
                suggest_dict[key] = suggest_list[i] if i < len(suggest_list) else ""
            return suggest_dict
    
    # 新增：调用本地qwen:0.5b模型生成/优化建议
    def _call_ollama_qwen(self, prompt):
        """
        调用本地部署的qwen:0.5b模型，生成/优化维修建议
        :param prompt: 提示词（包含故障信息、匹配建议、关键词）
        :return: 模型生成的结构化建议字符串
        """
        try:
            # 调用本地ollama模型（不联网，不消耗个人密钥/算力，企业部署后用自身服务器资源）
            response = self.ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "你是工业设备故障维修建议专家，仅输出维修建议，严格按照【应急处理】【根本修复】【预防措施】【备件清单】的格式组织语言，语言简洁专业，贴合工业场景，不添加任何多余解释，每条建议不超过20字。"},
                    {"role": "user", "content": prompt}
                ]
            )
            # 返回模型生成的内容（去除多余空格/换行）
            return response["message"]["content"].strip()
        except Exception as e:
            warnings.warn(f"本地qwen:0.5b模型调用失败：{str(e)}，将使用默认建议格式")
            return ""
    
    def get_suggestions(self, text1, text2, meta1, meta2, similarity_threshold=0.5, top_k=3):
        """
        根据故障文本匹配，获取维修建议（集成本地qwen:0.5b模型）
        逻辑：1. 有知识库数据→匹配建议+关键词，喂给模型优化格式；2. 无数据→调用模型兜底生成建议
        :param text1: 输入故障文本1
        :param text2: 输入故障文本2（可选，为空则仅匹配text1）
        :param meta1: 输入设备元数据1
        :param meta2: 输入设备元数据2（可选）
        :param similarity_threshold: 相似度阈值（低于阈值不返回建议）
        :param top_k: 取Top k条匹配结果
        :return: 结构化维修建议列表
        """
        # 1. 获取知识库所有有效数据
        kb_df = self.kb.get_data(valid_only=True)
        
        # 场景1：知识库无有效数据→调用本地qwen:0.5b模型兜底生成建议
        if kb_df.empty:
            warnings.warn("知识库无有效数据，将调用本地qwen:0.5b模型生成维修建议")
            # 构造模型提示词（包含故障文本、设备元数据，贴合工业场景）
            prompt = f"故障文本：{text1}；设备元数据：{meta1}；请生成工业设备维修建议，严格按照指定格式输出，语言简洁专业。"
            if text2.strip():
                prompt += f"补充故障文本：{text2}；补充设备元数据：{meta2}。"
            # 调用模型生成建议
            model_suggest = self._call_ollama_qwen(prompt)
            if not model_suggest:
                warnings.warn("模型生成建议失败，无有效维修建议")
                return []
            # 解析模型生成的建议，转为结构化格式
            suggest_dict = self._parse_suggest(model_suggest)
            valid_suggest = {k: v for k, v in suggest_dict.items() if self._validate_suggest(v)}
            if not valid_suggest:
                warnings.warn("模型生成的建议无效，无有效维修建议")
                return []
            # 封装为结构化返回格式（与有数据场景统一）
            final_suggests = []
            seen_suggests = set()
            for priority_key in self.priority:
                suggest_content = valid_suggest.get(priority_key, "")
                if suggest_content and suggest_content not in seen_suggests:
                    final_suggests.append({
                        "priority": priority_key,
                        "content": suggest_content,
                        "match_info": "本地qwen:0.5b模型生成（知识库无数据）",
                        "applicable_device": meta1 if meta1.strip() else meta2
                    })
                    seen_suggests.add(suggest_content)
            return final_suggests
        
        # 场景2：知识库有数据→匹配相似故障，提取建议+关键词，喂给模型优化格式
        # 2. 匹配相似故障（简化：基于文本包含关系+相似度，不重复计算）
        # 优先匹配故障文本1
        match_df = kb_df[kb_df["fault_text1"].str.contains(text1.split("，")[0], na=False)]
        # 补充匹配故障文本2（若有）
        if text2.strip():
            match_df2 = kb_df[kb_df["fault_text2"].str.contains(text2.split("，")[0], na=False)]
            match_df = pd.concat([match_df, match_df2]).drop_duplicates(subset=["id"])
        
        # 过滤低于阈值的数据
        match_df = match_df[match_df["similarity"] >= similarity_threshold].sort_values("similarity", ascending=False).head(top_k)
        if match_df.empty:
            warnings.warn(f"未匹配到相似度≥{similarity_threshold}的故障案例，调用本地qwen:0.5b模型生成建议")
            # 无匹配案例，调用模型兜底（同场景1逻辑）
            prompt = f"故障文本：{text1}；设备元数据：{meta1}；请生成工业设备维修建议，严格按照指定格式输出，语言简洁专业。"
            if text2.strip():
                prompt += f"补充故障文本：{text2}；补充设备元数据：{meta2}。"
            model_suggest = self._call_ollama_qwen(prompt)
            if not model_suggest:
                warnings.warn("模型生成建议失败，无有效维修建议")
                return []
            suggest_dict = self._parse_suggest(model_suggest)
            valid_suggest = {k: v for k, v in suggest_dict.items() if self._validate_suggest(v)}
            if not valid_suggest:
                warnings.warn("模型生成的建议无效，无有效维修建议")
                return []
            final_suggests = []
            seen_suggests = set()
            for priority_key in self.priority:
                suggest_content = valid_suggest.get(priority_key, "")
                if suggest_content and suggest_content not in seen_suggests:
                    final_suggests.append({
                        "priority": priority_key,
                        "content": suggest_content,
                        "match_info": "本地qwen:0.5b模型生成（无匹配案例）",
                        "applicable_device": meta1 if meta1.strip() else meta2
                    })
                    seen_suggests.add(suggest_content)
            return final_suggests
        
        # 3. 提取匹配到的建议、关键词，构造提示词，喂给模型优化格式
        # 提取所有匹配建议和关键词
        match_suggests = []
        match_keywords = []
        for _, row in match_df.iterrows():
            # 提取匹配建议
            suggest_dict = self._parse_suggest(row["repair_suggest"])
            valid_suggest = {k: v for k, v in suggest_dict.items() if self._validate_suggest(v)}
            if valid_suggest:
                match_suggests.append(str(valid_suggest))
            # 提取关键词（故障文本+设备元数据中的核心词）
            keywords = text1.split("，") + text2.split("，") + row["device_meta"].split("|")
            match_keywords.extend([k.strip() for k in keywords if k.strip()])
        # 去重关键词和建议
        match_suggests = list(set(match_suggests))[:top_k]  # 取前3条不重复建议
        match_keywords = list(set(match_keywords))[:10]  # 取前10个不重复关键词
        
        # 构造模型提示词（让模型基于匹配建议和关键词，优化格式、组织语言）
        prompt = f"故障文本：{text1}；补充故障文本：{text2}；设备元数据：{meta1}；" \
                 f"匹配到的维修建议：{','.join(match_suggests)}；相关关键词：{','.join(match_keywords)}；" \
                 f"请基于这些信息，按照【应急处理】【根本修复】【预防措施】【备件清单】的格式组织语言，生成简洁专业的工业维修建议，不添加多余解释，每条建议不超过20字。" 
        
        # 调用本地qwen:0.5b模型，优化建议格式
        optimized_suggest = self._call_ollama_qwen(prompt)
        if not optimized_suggest:
            warnings.warn("模型优化建议失败，将使用原始匹配建议")
            # 模型调用失败，使用原始匹配建议
            all_suggests = []
            for _, row in match_df.iterrows():
                suggest_dict = self._parse_suggest(row["repair_suggest"])
                valid_suggest = {k: v for k, v in suggest_dict.items() if self._validate_suggest(v)}
                if not valid_suggest:
                    continue
                structured_suggest = {
                    "match_id": row["id"],
                    "match_fault_text1": row["fault_text1"],
                    "match_fault_text2": row["fault_text2"],
                    "match_similarity": row["similarity"],
                    "device_meta": row["device_meta"],
                    "suggestions": valid_suggest,
                    "update_time": row["update_time"]
                }
                all_suggests.append(structured_suggest)
            # 按优先级排序，去重（原始逻辑）
            final_suggests = []
            seen_suggests = set()
            for suggest in all_suggests:
                for priority_key in self.priority:
                    suggest_content = suggest["suggestions"].get(priority_key, "")
                    if suggest_content and suggest_content not in seen_suggests:
                        final_suggests.append({
                            "priority": priority_key,
                            "content": suggest_content,
                            "match_info": f"匹配案例ID：{suggest['match_id']}（相似度：{suggest['match_similarity']:.4f}）",
                            "applicable_device": suggest["device_meta"]
                        })
                        seen_suggests.add(suggest_content)
            return final_suggests
        
        # 4. 解析模型优化后的建议，转为结构化格式（与原始逻辑统一）
        optimized_dict = self._parse_suggest(optimized_suggest)
        valid_optimized = {k: v for k, v in optimized_dict.items() if self._validate_suggest(v)}
        if not valid_optimized:
            warnings.warn("模型优化后的建议无效，将使用原始匹配建议")
            #  fallback到原始匹配建议
            all_suggests = []
            for _, row in match_df.iterrows():
                suggest_dict = self._parse_suggest(row["repair_suggest"])
                valid_suggest = {k: v for k, v in suggest_dict.items() if self._validate_suggest(v)}
                if not valid_suggest:
                    continue
                structured_suggest = {
                    "match_id": row["id"],
                    "match_fault_text1": row["fault_text1"],
                    "match_fault_text2": row["fault_text2"],
                    "match_similarity": row["similarity"],
                    "device_meta": row["device_meta"],
                    "suggestions": valid_suggest,
                    "update_time": row["update_time"]
                }
                all_suggests.append(structured_suggest)
            final_suggests = []
            seen_suggests = set()
            for suggest in all_suggests:
                for priority_key in self.priority:
                    suggest_content = suggest["suggestions"].get(priority_key, "")
                    if suggest_content and suggest_content not in seen_suggests:
                        final_suggests.append({
                            "priority": priority_key,
                            "content": suggest_content,
                            "match_info": f"匹配案例ID：{suggest['match_id']}（相似度：{suggest['match_similarity']:.4f}）",
                            "applicable_device": suggest["device_meta"]
                        })
                        seen_suggests.add(suggest_content)
            return final_suggests
        
        # 5. 封装模型优化后的结构化建议（关联匹配信息）
        final_suggests = []
        seen_suggests = set()
        # 获取匹配度最高的案例信息（用于展示）
        top_match = match_df.iloc[0]
        for priority_key in self.priority:
            suggest_content = valid_optimized.get(priority_key, "")
            if suggest_content and suggest_content not in seen_suggests:
                final_suggests.append({
                    "priority": priority_key,
                    "content": suggest_content,
                    "match_info": f"本地qwen:0.5b优化（匹配案例ID：{top_match['id']}，相似度：{top_match['similarity']:.4f}）",
                    "applicable_device": top_match["device_meta"]
                })
                seen_suggests.add(suggest_content)
        
        return final_suggests
    
    def print_suggestions(self, suggestions):
        """打印结构化维修建议（用于展示/接口输出）"""
        if not suggestions:
            print("❌ 未生成有效维修建议")
            return
        
        print("\n" + "="*60)
        print("📋 故障维修建议（按优先级排序）")
        print("="*60)
        for idx, suggest in enumerate(suggestions, 1):
            print(f"{idx}. 【{suggest['priority']}】")
            print(f"   建议内容：{suggest['content']}")
            print(f"   匹配信息：{suggest['match_info']}")
            print(f"   适用设备：{suggest['applicable_device']}")
            print("-"*60)
    
    def export_suggestions(self, suggestions, save_path=None):
        """导出维修建议为JSON文件（企业运维可用）"""
        if not suggestions:
            warnings.warn("无有效维修建议可导出")
            return
        
        if save_path is None:
            save_path = os.path.join(BASE_PATH, "repair_suggestions.json")
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(suggestions, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 维修建议已导出到：{save_path}")
        return save_path

# 测试代码（可注释）
if __name__ == "__main__":
    from knowledge_base import FaultKnowledgeBase
    # 初始化知识库
    kb = FaultKnowledgeBase()
    # 初始化建议生成器
    generator = RepairSuggestionGenerator(kb)
    
    # 测试生成建议（两种场景可切换测试）
    text1 = "液压缸泄漏，无法正常加压"
    text2 = ""
    meta1 = "液压泵|型号A|左侧"
    meta2 = ""
    suggestions = generator.get_suggestions(text1, text2, meta1, meta2, similarity_threshold=0.5)
    generator.print_suggestions(suggestions)
    
    # 关闭知识库连接
    kb.close()