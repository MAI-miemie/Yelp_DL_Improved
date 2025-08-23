"""
模型训练模块 - 大样本版本
包含数据加载、模型训练、评估等完整流程，支持大规模数据集
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
from tqdm import tqdm
import wandb
import gc
import psutil
import os

# 导入自定义模块
from data_loader import YelpDataLoader
from models.bert_model import TextFeatureExtractor, create_text_features
from models.recommendation import create_recommendation_model, HybridRecommender

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeScaleModelTrainer:
    """大规模模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 内存管理
        self.memory_threshold = config.get('memory_threshold', 0.8)  # 内存使用阈值
        self.chunk_size = config.get('chunk_size', 10000)  # 数据分片大小
        
        # 初始化wandb（可选）
        if config.get('use_wandb', False):
            wandb.init(project=config.get('project_name', 'yelp-recommendation-large'))
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rate': []
        }
    
    def load_large_scale_data(self, data_paths: Dict) -> Dict:
        """
        加载大规模数据
        
        Args:
            data_paths: 数据文件路径字典
            
        Returns:
            处理后的数据字典
        """
        logger.info("开始加载大规模数据...")
        
        # 检查内存使用情况
        self._check_memory_usage()
        
        # 初始化数据加载器
        loader = YelpDataLoader()
        
        # 分片加载数据
        business_df = self._load_data_in_chunks(
            loader.load_business_data, 
            data_paths.get('business', 'data/yelp_business.csv')
        )
        logger.info(f"商户数据加载完成，形状: {business_df.shape}")
        
        reviews_df = self._load_data_in_chunks(
            loader.load_reviews_data,
            data_paths.get('reviews', 'data/reviews_of_restaurants.txt')
        )
        logger.info(f"评论数据加载完成，形状: {reviews_df.shape}")
        
        users_df = self._load_data_in_chunks(
            loader.load_users_data,
            data_paths.get('users', 'data/users.txt')
        )
        logger.info(f"用户数据加载完成，形状: {users_df.shape}")
        
        # 创建交互矩阵（分片处理）
        interaction_matrix, mappings = self._create_large_interaction_matrix(
            reviews_df, business_df, users_df
        )
        logger.info(f"交互矩阵创建完成，形状: {interaction_matrix.shape}")
        
        # 提取文本特征（分片处理）
        text_features = self._extract_text_features_in_chunks(reviews_df)
        logger.info(f"文本特征提取完成，形状: {text_features.shape}")
        
        return {
            'business_df': business_df,
            'reviews_df': reviews_df,
            'users_df': users_df,
            'interaction_matrix': interaction_matrix,
            'mappings': mappings,
            'text_features': text_features
        }
    
    def _load_data_in_chunks(self, load_func, file_path: str) -> pd.DataFrame:
        """分片加载数据"""
        try:
            # 尝试直接加载
            return load_func(file_path)
        except MemoryError:
            logger.warning(f"内存不足，使用分片加载: {file_path}")
            return self._chunk_load_data(file_path)
    
    def _chunk_load_data(self, file_path: str) -> pd.DataFrame:
        """分片加载数据"""
        if file_path.endswith('.csv'):
            # CSV文件分片读取
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                chunks.append(chunk)
                self._check_memory_usage()
            return pd.concat(chunks, ignore_index=True)
        elif file_path.endswith('.txt'):
            # JSON文件分片读取
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % self.chunk_size == 0:
                        self._check_memory_usage()
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            return pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    def _create_large_interaction_matrix(self, reviews_df: pd.DataFrame, 
                                       business_df: pd.DataFrame, 
                                       users_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """创建大规模交互矩阵"""
        logger.info("创建大规模交互矩阵...")
        
        # 创建用户和商户的映射
        user_mapping = {user_id: idx for idx, user_id in enumerate(users_df['user_id'].unique())}
        business_mapping = {business_id: idx for idx, business_id in enumerate(business_df['business_id'].unique())}
        
        # 创建交互矩阵
        num_users = len(user_mapping)
        num_businesses = len(business_mapping)
        
        # 使用稀疏矩阵来节省内存
        from scipy.sparse import csr_matrix
        
        # 收集交互数据
        user_indices = []
        business_indices = []
        ratings = []
        
        for _, review in reviews_df.iterrows():
            user_id = review['user_id']
            business_id = review['business_id']
            rating = review['stars']
            
            if user_id in user_mapping and business_id in business_mapping:
                user_indices.append(user_mapping[user_id])
                business_indices.append(business_mapping[business_id])
                ratings.append(rating)
        
        # 创建稀疏矩阵
        interaction_matrix = csr_matrix(
            (ratings, (user_indices, business_indices)),
            shape=(num_users, num_businesses)
        )
        
        mappings = {
            'user_mapping': user_mapping,
            'business_mapping': business_mapping
        }
        
        return interaction_matrix, mappings
    
    def _extract_text_features_in_chunks(self, reviews_df: pd.DataFrame) -> np.ndarray:
        """分片提取文本特征"""
        logger.info("分片提取文本特征...")
        
        text_features = []
        chunk_size = self.config.get('text_chunk_size', 1000)
        
        for i in range(0, len(reviews_df), chunk_size):
            chunk = reviews_df.iloc[i:i+chunk_size]
            chunk_texts = chunk['text'].tolist()
            
            # 提取文本特征（简化版本，避免BERT加载）
            chunk_features = self._extract_simple_text_features(chunk_texts)
            text_features.append(chunk_features)
            
            self._check_memory_usage()
        
        return np.vstack(text_features)
    
    def _extract_simple_text_features(self, texts: List[str]) -> np.ndarray:
        """提取简单文本特征（避免BERT加载）"""
        # 使用简单的TF-IDF特征
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=768,  # 与BERT维度一致
            stop_words=None,
            ngram_range=(1, 2)
        )
        
        # 处理空文本
        texts = [text if text else "无评论" for text in texts]
        
        features = vectorizer.fit_transform(texts).toarray()
        return features
    
    def prepare_large_scale_training_data(self, data: Dict) -> Tuple[DataLoader, DataLoader]:
        """
        准备大规模训练数据
        
        Args:
            data: 数据字典
            
        Returns:
            训练和验证数据加载器
        """
        logger.info("准备大规模训练数据...")
        
        interaction_matrix = data['interaction_matrix']
        
        # 转换为密集矩阵（如果内存允许）
        if hasattr(interaction_matrix, 'toarray'):
            interaction_matrix = interaction_matrix.toarray()
        
        # 创建训练样本
        train_data, val_data = self._create_training_samples(interaction_matrix)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.get('batch_size', 256),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.get('batch_size', 256),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _create_training_samples(self, interaction_matrix: np.ndarray) -> Tuple[TensorDataset, TensorDataset]:
        """创建训练样本"""
        # 获取正样本（有交互的用户-商户对）
        positive_indices = np.where(interaction_matrix > 0)
        positive_users = positive_indices[0]
        positive_items = positive_indices[1]
        positive_ratings = interaction_matrix[positive_indices]
        
        # 创建负样本（无交互的用户-商户对）
        negative_users, negative_items = self._sample_negative_pairs(
            interaction_matrix, len(positive_users)
        )
        negative_ratings = np.zeros(len(negative_users))
        
        # 合并正负样本
        all_users = np.concatenate([positive_users, negative_users])
        all_items = np.concatenate([positive_items, negative_items])
        all_ratings = np.concatenate([positive_ratings, negative_ratings])
        
        # 转换为张量
        users_tensor = torch.LongTensor(all_users)
        items_tensor = torch.LongTensor(all_items)
        ratings_tensor = torch.FloatTensor(all_ratings)
        
        # 分割训练和验证集
        train_size = int(0.8 * len(users_tensor))
        
        train_dataset = TensorDataset(
            users_tensor[:train_size],
            items_tensor[:train_size],
            ratings_tensor[:train_size]
        )
        
        val_dataset = TensorDataset(
            users_tensor[train_size:],
            items_tensor[train_size:],
            ratings_tensor[train_size:]
        )
        
        return train_dataset, val_dataset
    
    def _sample_negative_pairs(self, interaction_matrix: np.ndarray, num_negative: int) -> Tuple[np.ndarray, np.ndarray]:
        """采样负样本对"""
        num_users, num_items = interaction_matrix.shape
        
        negative_users = []
        negative_items = []
        
        for _ in range(num_negative):
            user = np.random.randint(0, num_users)
            item = np.random.randint(0, num_items)
            
            # 确保是负样本
            while interaction_matrix[user, item] > 0:
                user = np.random.randint(0, num_users)
                item = np.random.randint(0, num_items)
            
            negative_users.append(user)
            negative_items.append(item)
        
        return np.array(negative_users), np.array(negative_items)
    
    def train_large_scale_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                               data: Dict) -> nn.Module:
        """
        训练大规模模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            data: 数据字典
            
        Returns:
            训练好的模型
        """
        logger.info("开始大规模模型训练...")
        
        # 创建模型
        num_users = len(data['mappings']['user_mapping'])
        num_items = len(data['mappings']['business_mapping'])
        
        model = create_recommendation_model(
            model_type=self.config.get('model_type', 'ncf'),
            num_users=num_users,
            num_items=num_items,
            embed_dim=self.config.get('embed_dim', 64),
            mlp_dims=self.config.get('mlp_dims', [128, 64, 32])
        ).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get('lr_step_size', 5),
            gamma=self.config.get('lr_gamma', 0.5)
        )
        
        # 训练循环
        num_epochs = self.config.get('epochs', 50)
        best_val_loss = float('inf')
        patience = self.config.get('patience', 10)
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss = self._validate_epoch(model, val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), self.output_dir / 'best_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            # 内存清理
            self._cleanup_memory()
        
        # 加载最佳模型
        model.load_state_dict(torch.load(self.output_dir / 'best_model.pth'))
        
        return model
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="训练"):
            users, items, ratings = batch
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)
            
            # 前向传播
            predictions = model(users, items)
            loss = criterion(predictions.squeeze(), ratings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> float:
        """验证一个epoch"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                users, items, ratings = batch
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                
                predictions = model(users, items)
                loss = criterion(predictions.squeeze(), ratings)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100
        if memory_percent > self.memory_threshold:
            logger.warning(f"内存使用率过高: {memory_percent:.2%}")
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_training_results(self, model: nn.Module, data: Dict):
        """保存训练结果"""
        # 保存模型
        torch.save(model.state_dict(), self.output_dir / 'final_model.pth')
        
        # 保存训练历史
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # 保存映射
        with open(self.output_dir / 'mappings.json', 'w') as f:
            json.dump(data['mappings'], f, indent=2)
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        logger.info(f"训练结果已保存到: {self.output_dir}")
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['train_loss'], label='训练损失')
        plt.plot(self.train_history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练和验证损失')
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('学习率变化')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

# 保持向后兼容
class ModelTrainer(LargeScaleModelTrainer):
    """向后兼容的模型训练器"""
    pass
