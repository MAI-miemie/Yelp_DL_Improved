"""
数据加载和预处理模块
负责加载原始数据、数据清洗、特征提取等
"""

import pandas as pd
import numpy as np
import json
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YelpDataLoader:
    """Yelp数据加载器"""
    
    def __init__(self, data_dir: str = "../data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.business_data = None
        self.reviews_data = None
        self.users_data = None
        
    def load_business_data(self, file_path: str = "../yelp_business.csv") -> pd.DataFrame:
        """
        加载商户数据
        
        Args:
            file_path: 商户数据文件路径
            
        Returns:
            处理后的商户数据DataFrame
        """
        logger.info("开始加载商户数据...")
        
        # 加载原始数据
        df = pd.read_csv(file_path)
        logger.info(f"原始商户数据形状: {df.shape}")
        
        # 数据清洗
        df = self._clean_business_data(df)
        
        # 特征提取
        df = self._extract_business_features(df)
        
        self.business_data = df
        logger.info(f"处理后商户数据形状: {df.shape}")
        
        return df
    
    def load_reviews_data(self, file_path: str = "../reviews_of_restaurants.txt") -> pd.DataFrame:
        """
        加载评论数据
        
        Args:
            file_path: 评论数据文件路径
            
        Returns:
            处理后的评论数据DataFrame
        """
        logger.info("开始加载评论数据...")
        
        # 读取评论数据
        reviews = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取评论数据"):
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(reviews)
        logger.info(f"原始评论数据形状: {df.shape}")
        
        # 数据清洗
        df = self._clean_reviews_data(df)
        
        self.reviews_data = df
        logger.info(f"处理后评论数据形状: {df.shape}")
        
        return df
    
    def load_users_data(self, file_path: str = "../users.txt") -> pd.DataFrame:
        """
        加载用户数据
        
        Args:
            file_path: 用户数据文件路径
            
        Returns:
            处理后的用户数据DataFrame
        """
        logger.info("开始加载用户数据...")
        
        # 读取用户数据
        users = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取用户数据"):
                try:
                    user = json.loads(line.strip())
                    users.append(user)
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(users)
        logger.info(f"原始用户数据形状: {df.shape}")
        
        # 数据清洗
        df = self._clean_users_data(df)
        
        self.users_data = df
        logger.info(f"处理后用户数据形状: {df.shape}")
        
        return df
    
    def _clean_business_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗商户数据"""
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        df['categories'] = df['categories'].fillna('')
        df['attributes'] = df['attributes'].fillna('{}')
        df['hours'] = df['hours'].fillna('{}')
        
        # 解析JSON字段
        df['attributes_dict'] = df['attributes'].apply(self._safe_json_parse)
        df['hours_dict'] = df['hours'].apply(self._safe_json_parse)
        
        # 提取关键属性
        df['accepts_credit_cards'] = df['attributes_dict'].apply(
            lambda x: x.get('BusinessAcceptsCreditCards', 'False') == 'True'
        )
        df['parking'] = df['attributes_dict'].apply(
            lambda x: 'BusinessParking' in x
        )
        df['wifi'] = df['attributes_dict'].apply(
            lambda x: x.get('WiFi', 'no') != 'no'
        )
        
        # 处理类别
        df['category_list'] = df['categories'].apply(
            lambda x: [cat.strip() for cat in str(x).split(',') if cat.strip()]
        )
        
        return df
    
    def _clean_reviews_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗评论数据"""
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        df = df.dropna(subset=['text', 'stars'])
        
        # 过滤有效评论
        df = df[df['text'].str.len() > 10]  # 评论长度至少10个字符
        df = df[df['stars'].between(1, 5)]  # 星级在1-5之间
        
        # 转换数据类型
        df['stars'] = df['stars'].astype(float)
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def _clean_users_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗用户数据"""
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值
        df['friends'] = df['friends'].fillna('[]')
        df['elite'] = df['elite'].fillna('[]')
        
        # 解析朋友列表
        df['friends_list'] = df['friends'].apply(self._safe_json_parse)
        df['elite_list'] = df['elite'].apply(self._safe_json_parse)
        
        # 计算用户特征
        df['friend_count'] = df['friends_list'].apply(len)
        df['elite_count'] = df['elite_list'].apply(len)
        df['review_count'] = df['review_count'].fillna(0).astype(int)
        
        return df
    
    def _extract_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取商户特征"""
        # 类别特征
        df['category_count'] = df['category_list'].apply(len)
        df['is_restaurant'] = df['categories'].str.contains('Restaurants', case=False)
        df['is_food'] = df['categories'].str.contains('Food', case=False)
        
        # 位置特征
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # 营业时间特征
        df['has_hours'] = df['hours_dict'].apply(lambda x: len(x) > 0)
        
        return df
    
    def _safe_json_parse(self, text: str) -> dict:
        """安全解析JSON字符串"""
        if pd.isna(text) or text == '':
            return {}
        try:
            if isinstance(text, str):
                return ast.literal_eval(text)
            return text
        except:
            return {}
    
    def create_interaction_matrix(self) -> Tuple[pd.DataFrame, Dict]:
        """
        创建用户-商户交互矩阵
        
        Returns:
            交互矩阵和映射字典
        """
        if self.reviews_data is None or self.business_data is None:
            raise ValueError("请先加载评论数据和商户数据")
        
        logger.info("创建用户-商户交互矩阵...")
        
        # 合并数据
        interactions = self.reviews_data.merge(
            self.business_data[['business_id', 'name', 'categories']], 
            on='business_id', 
            how='inner'
        )
        
        # 创建用户和商户的映射
        user_mapping = {user_id: idx for idx, user_id in enumerate(interactions['user_id'].unique())}
        business_mapping = {business_id: idx for idx, business_id in enumerate(interactions['business_id'].unique())}
        
        # 创建交互矩阵
        interaction_matrix = interactions.pivot_table(
            index='user_id', 
            columns='business_id', 
            values='stars', 
            fill_value=0
        )
        
        logger.info(f"交互矩阵形状: {interaction_matrix.shape}")
        
        return interaction_matrix, {
            'user_mapping': user_mapping,
            'business_mapping': business_mapping,
            'interactions': interactions
        }
    
    def save_processed_data(self, output_dir: str = "../data/processed"):
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.business_data is not None:
            self.business_data.to_csv(output_path / "business_processed.csv", index=False)
        
        if self.reviews_data is not None:
            self.reviews_data.to_csv(output_path / "reviews_processed.csv", index=False)
        
        if self.users_data is not None:
            self.users_data.to_csv(output_path / "users_processed.csv", index=False)
        
        logger.info(f"数据已保存到: {output_path}")

def main():
    """主函数 - 数据加载和预处理流程"""
    # 初始化数据加载器
    loader = YelpDataLoader()
    
    # 加载数据
    business_df = loader.load_business_data()
    reviews_df = loader.load_reviews_data()
    users_df = loader.load_users_data()
    
    # 创建交互矩阵
    interaction_matrix, mappings = loader.create_interaction_matrix()
    
    # 保存处理后的数据
    loader.save_processed_data()
    
    # 保存交互矩阵
    interaction_matrix.to_csv("../data/processed/interaction_matrix.csv")
    
    # 保存映射信息
    with open("../data/processed/mappings.json", 'w') as f:
        json.dump({
            'user_mapping': mappings['user_mapping'],
            'business_mapping': mappings['business_mapping']
        }, f)
    
    logger.info("数据加载和预处理完成！")

if __name__ == "__main__":
    main()
