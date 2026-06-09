import os
import sqlite3
import pandas as pd
import time
import warnings


# 所有路径统一为 D:\industrial_nlp_project
BASE_PATH = "D:\\industrial_nlp_project"
# 知识库数据库路径（SQLite，无需额外部署）
DB_PATH = os.path.join(BASE_PATH, "fault_knowledge_base.db")

class FaultKnowledgeBase:
    def __init__(self):
        """初始化知识库，创建数据库和表（不存在则创建）"""
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()  # 创建故障知识库表
        self._create_version_table()  # 创建版本管理表
    
    def _create_table(self):
        """创建故障知识库核心表（字段适配现有数据格式）"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS fault_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fault_text1 TEXT NOT NULL,
            fault_text2 TEXT NOT NULL,
            similarity FLOAT NOT NULL DEFAULT 0.0,
            repair_suggest TEXT NOT NULL DEFAULT "",
            device_meta TEXT NOT NULL DEFAULT "",
            update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            is_valid BOOLEAN NOT NULL DEFAULT 1
        )
        """
        self.cursor.execute(create_sql)
        self.conn.commit()
    
    def _create_version_table(self):
        """创建版本管理表，用于数据更新回滚"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS data_version (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            update_desc TEXT NOT NULL DEFAULT "",
            data_count INTEGER NOT NULL DEFAULT 0
        )
        """
        self.cursor.execute(create_sql)
        self.conn.commit()
        # 初始化版本（若为空）
        if not self._get_latest_version():
            self._add_version(update_desc="初始化知识库")
    
    def _add_version(self, update_desc=""):
        """新增版本记录"""
        data_count = self.get_data_count()
        self.cursor.execute(
            "INSERT INTO data_version (update_desc, data_count) VALUES (?, ?)",
            (update_desc, data_count)
        )
        self.conn.commit()
    
    def _get_latest_version(self):
        """获取最新版本记录"""
        self.cursor.execute("SELECT * FROM data_version ORDER BY version_id DESC LIMIT 1")
        return self.cursor.fetchone()
    
    def add_data(self, df, update_desc="新增数据"):
        """
        新增数据到知识库（支持批量）
        :param df: DataFrame，需包含fault_text1, fault_text2等标准化字段（来自document_parser）
        :param update_desc: 数据更新描述
        """
        if df.empty:
            warnings.warn("无有效数据可添加到知识库")
            return
        
        # 数据校验（过滤无效数据，但不中断）
        df = df[(df["fault_text1"].str.strip() != "") & (df["fault_text2"].str.strip() != "")].reset_index(drop=True)
        if df.empty:
            warnings.warn("过滤后无有效数据可添加")
            return
        
        # 批量插入数据
        data_list = []
        for _, row in df.iterrows():
            data_list.append((
                row["fault_text1"],
                row["fault_text2"],
                round(row["similarity"], 4),
                row["repair_suggest"],
                row["device_meta"],
                1  # is_valid默认1（有效）
            ))
        self.cursor.executemany(
            "INSERT INTO fault_data (fault_text1, fault_text2, similarity, repair_suggest, device_meta) VALUES (?, ?, ?, ?, ?)",
            data_list
        )
        self.conn.commit()
        self._add_version(update_desc=update_desc)
        print(f"✅ 成功添加{len(data_list)}条故障数据到知识库")
        self.cursor.executemany("""
            INSERT INTO fault_data (fault_text1, fault_text2, similarity, repair_suggest, device_meta, is_valid)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data_list)
        self.conn.commit()
        # 新增版本记录
        self._add_version(update_desc=update_desc)
        print(f"✅ 成功添加 {len(df)} 条数据到知识库，当前版本：{self._get_latest_version()[0]}")
    
    def update_data(self, id_list, new_data, update_desc="更新数据"):
        """
        更新知识库中的数据（按id更新）
        :param id_list: 要更新的数据id列表
        :param new_data: DataFrame，与id_list一一对应，包含要更新的字段
        """
        if not id_list or new_data.empty or len(id_list) != len(new_data):
            warnings.warn("更新参数不合法，无法更新数据")
            return
        
        # 批量更新
        for idx, data_id in enumerate(id_list):
            row = new_data.iloc[idx]
            # 仅更新非空字段（妥协逻辑，避免覆盖原有有效数据）
            update_fields = []
            update_values = []
            if row["fault_text1"].strip():
                update_fields.append("fault_text1 = ?")
                update_values.append(row["fault_text1"])
            if row["fault_text2"].strip():
                update_fields.append("fault_text2 = ?")
                update_values.append(row["fault_text2"])
            if isinstance(row["similarity"], (int, float)):
                update_fields.append("similarity = ?")
                update_values.append(round(row["similarity"], 4))
            if row["repair_suggest"].strip():
                update_fields.append("repair_suggest = ?")
                update_values.append(row["repair_suggest"])
            if row["device_meta"].strip():
                update_fields.append("device_meta = ?")
                update_values.append(row["device_meta"])
            
            if not update_fields:
                continue
            
            update_sql = f"UPDATE fault_data SET {', '.join(update_fields)} WHERE id = ?"
            update_values.append(data_id)
            self.cursor.execute(update_sql, update_values)
        
        self.conn.commit()
        self._add_version(update_desc=update_desc)
        print(f"✅ 成功更新 {len(id_list)} 条数据，当前版本：{self._get_latest_version()[0]}")
    
    def delete_data(self, id_list, update_desc="删除数据"):
        """按id删除数据（逻辑删除，设置is_valid=0）"""
        if not id_list:
            warnings.warn("无有效id可删除")
            return
        
        # 逻辑删除（不物理删除，便于回滚）
        self.cursor.executemany("UPDATE fault_data SET is_valid = 0 WHERE id = ?", [(id,) for id in id_list])
        self.conn.commit()
        self._add_version(update_desc=update_desc)
        print(f"✅ 成功逻辑删除 {len(id_list)} 条数据，当前版本：{self._get_latest_version()[0]}")
    
    def get_data(self, valid_only=True, version_id=None):
        """
        获取知识库数据
        :param valid_only: 是否只获取有效数据（is_valid=1）
        :param version_id: 按版本获取（None则获取最新）
        :return: DataFrame
        """
        # 按版本获取（简化：版本对应数据量，获取对应时间范围内的数据）
        if version_id:
            self.cursor.execute("SELECT update_time FROM data_version WHERE version_id = ?", (version_id,))
            version_time = self.cursor.fetchone()
            if not version_time:
                warnings.warn(f"版本 {version_id} 不存在，获取最新数据")
                version_time = self._get_latest_version()[2]
            else:
                version_time = version_time[0]
            where_clause = f"update_time <= '{version_time}' AND is_valid = 1" if valid_only else f"update_time <= '{version_time}'"
        else:
            where_clause = "is_valid = 1" if valid_only else ""
        
        sql = "SELECT id, fault_text1, fault_text2, similarity, repair_suggest, device_meta, update_time FROM fault_data"
        if where_clause:
            sql += f" WHERE {where_clause}"
        
        df = pd.read_sql_query(sql, self.conn)
        return df
    
    def export_to_csv(self, save_path=None, valid_only=True):
        """导出知识库数据为CSV，用于模型训练"""
        if save_path is None:
            save_path = os.path.join(BASE_PATH, "knowledge_base_export.csv")
        
        df = self.get_data(valid_only=valid_only)
        if df.empty:
            warnings.warn("知识库无有效数据可导出")
            return
        
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"✅ 知识库数据已导出到：{save_path}，共 {len(df)} 条数据")
        return save_path
    
    def get_data_count(self, valid_only=True):
        """获取数据总量"""
        where_clause = "WHERE is_valid = 1" if valid_only else ""
        self.cursor.execute(f"SELECT COUNT(*) FROM fault_data {where_clause}")
        return self.cursor.fetchone()[0]
    def get_all_faults(self):
        """获取所有有效故障文本+维修方案（适配RAG引擎向量检索）"""
        # 只查询有效数据，与RAG检索场景匹配
        self.cursor.execute("""
            SELECT fault_text1, repair_suggest, device_meta 
            FROM fault_data 
            WHERE is_valid = 1
        """)
        # 获取查询结果
        fault_records = self.cursor.fetchall()
        # 整理为RAG引擎可直接使用的字典列表
        all_faults = []
        for record in fault_records:
            fault_text, repair_suggest, device_meta = record
            # 拼接设备元数据，丰富故障文本信息（提升RAG检索精度）
            full_fault_text = f"{fault_text}【设备信息：{device_meta}】" if device_meta else fault_text
            all_faults.append({
                "text": full_fault_text,  # RAG检索用的故障文本（含元数据）
                "solution": repair_suggest  # 对应的维修方案（RAG增强生成用）
            })
        return all_faults
    def get_data_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM fault_data WHERE is_valid=1")
        return self.cursor.fetchone()[0]

    def get_all_faults(self):
        self.cursor.execute("SELECT fault_text1 as text, repair_suggest as solution FROM fault_data WHERE is_valid=1")
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def get_data(self, valid_only=True):
        where_clause = "WHERE is_valid=1" if valid_only else ""
        self.cursor.execute(f"SELECT * FROM fault_data {where_clause}")
        columns = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame([dict(zip(columns, row)) for row in self.cursor.fetchall()])
        return df

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

# 测试代码（修改后，无test.csv也能运行，有明确终端输出）
if __name__ == "__main__":
    kb = FaultKnowledgeBase()
    print("="*50)
    print("✅ 知识库初始化成功！")
    print(f"当前知识库版本：{kb._get_latest_version()[0]}")
    print(f"当前有效数据量：{kb.get_data_count()}")
    print(f"📁 数据库文件路径：{DB_PATH}")
    # 新增验证代码 ↓↓↓
    all_faults = kb.get_all_faults()
    print(f"✅ 可检索故障案例数：{len(all_faults)}")
    if all_faults:
        print(f"示例故障案例：{all_faults[0]['text']}")
        print(f"示例维修方案：{all_faults[0]['solution']}")
    # 新增验证代码 ↑↑↑
    print("="*50)
    kb.close()