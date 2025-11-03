import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os

warnings.filterwarnings("ignore")

class TwoTowerModel(nn.Module):
    """Two-Tower Architecture for Query-Product Matching"""
    
    def __init__(self, model_name='distilbert-base-uncased', embedding_dim=256):
        super(TwoTowerModel, self).__init__()
        
        # Query tower
        self.query_encoder = AutoModel.from_pretrained(model_name)
        
        # Product tower
        self.product_encoder = AutoModel.from_pretrained(model_name)
        
        # Projection layers to ensure same embedding dimension
        self.query_projection = nn.Linear(self.query_encoder.config.hidden_size, embedding_dim)
        self.product_projection = nn.Linear(self.product_encoder.config.hidden_size, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def encode_query(self, input_ids, attention_mask):
        """Encode query using query tower"""
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        query_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        query_embedding = self.dropout(query_embedding)
        query_embedding = self.query_projection(query_embedding)  # [batch_size, embedding_dim]
        # L2 normalize
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        return query_embedding
    
    def encode_product(self, input_ids, attention_mask):
        """Encode product using product tower"""
        outputs = self.product_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        product_embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        product_embedding = self.dropout(product_embedding)
        product_embedding = self.product_projection(product_embedding)  # [batch_size, embedding_dim]
        # L2 normalize
        product_embedding = torch.nn.functional.normalize(product_embedding, p=2, dim=1)
        return product_embedding

class TwoTowerEvaluator:
    """Evaluator for Two-Tower model performance"""
    
    def __init__(self, model_path, test_data_path, embedding_dim=256):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = TwoTowerModel(model_name='distilbert-base-uncased', embedding_dim=embedding_dim)
        
        # Load model weights
        model_state = torch.load(os.path.join(model_path, 'model.pth'), map_location=self.device)
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        # Load evaluation data
        self.eval_data = pd.read_csv(test_data_path)
        
        # Load product corpus
        try:
            df_products = pd.read_parquet('dataset/shopping_queries_dataset_products.parquet')
            # Create product text mapping
            product_cols = ['product_title', 'product_brand', 'product_color', 'product_description', 'product_bullet_point']
            for col in product_cols:
                df_products[col] = df_products[col].fillna('')
            
            df_products['product_text'] = (
                df_products['product_title'] + ' ' +
                df_products['product_description'].str[:200] + ' ' +
                df_products['product_brand'] + ' ' +
                df_products['product_bullet_point'].str[:100]
            )
            df_products['product_text'] = df_products['product_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
            df_products['product_text'] = df_products['product_text'].str[:400]
            
            self.product_corpus = df_products[['product_id', 'product_text', 'product_title']].copy()
            
        except Exception as e:
            print(f"Warning: Could not load product corpus: {e}")
            self.product_corpus = None
        
        # Product embeddings cache
        self.product_embeddings = None
        self.product_ids = None
    
    def encode_text(self, texts, is_query=True, batch_size=32):
        """Encode texts using the appropriate tower"""
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {'queries' if is_query else 'products'}"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Encode using appropriate tower
                if is_query:
                    batch_embeddings = self.model.encode_query(input_ids, attention_mask)
                else:
                    batch_embeddings = self.model.encode_product(input_ids, attention_mask)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def build_product_index(self):
        """Build product embeddings index for fast retrieval"""
        if self.product_corpus is None:
            raise ValueError("Product corpus not loaded")
        
        print("Building product embeddings index...")
        
        # Get unique products from evaluation data
        eval_product_ids = set(self.eval_data['product_id'].unique())
        
        # Filter product corpus to only include products in evaluation
        relevant_products = self.product_corpus[self.product_corpus['product_id'].isin(eval_product_ids)].copy()
        
        print(f"Encoding {len(relevant_products)} products...")
        
        # Encode products
        product_texts = relevant_products['product_text'].tolist()
        self.product_embeddings = self.encode_text(product_texts, is_query=False)
        self.product_ids = relevant_products['product_id'].tolist()
        
        print(f"Product index built with {len(self.product_ids)} products")
    
    def search_products(self, query, top_k=10):
        """Search for products given a query"""
        if self.product_embeddings is None:
            raise ValueError("Product index not built. Call build_product_index() first.")
        
        # Encode query
        query_embedding = self.encode_text([query], is_query=True)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            product_id = self.product_ids[idx]
            similarity = similarities[idx]
            results.append((product_id, similarity))
        
        return results
    
    def calculate_metrics(self, k_values=[1, 5, 10, 20]):
        """Calculate comprehensive evaluation metrics"""
        if self.product_embeddings is None:
            raise ValueError("Product index not built. Call build_product_index() first.")
        
        # Group evaluation data by query
        query_groups = self.eval_data.groupby('query')
        
        metrics_by_k = {k: {'precision': [], 'recall': [], 'ndcg': [], 'mrr': []} for k in k_values}
        
        print(f"Evaluating on {len(query_groups)} unique queries...")
        
        for query, group in tqdm(query_groups, desc="Calculating metrics"):
            # Get ground truth relevant products
            relevant_products = set(group[group['esci_label'] == 'E']['product_id'].tolist())
            
            if not relevant_products:
                continue
            
            # Search for products
            max_k = max(k_values)
            results = self.search_products(query, top_k=max_k)
            
            # Calculate metrics for each k
            for k in k_values:
                top_k_results = results[:k]
                retrieved_products = [pid for pid, _ in top_k_results]
                
                # Precision@k
                relevant_retrieved = len(set(retrieved_products) & relevant_products)
                precision = relevant_retrieved / k if k > 0 else 0
                metrics_by_k[k]['precision'].append(precision)
                
                # Recall@k
                recall = relevant_retrieved / len(relevant_products) if relevant_products else 0
                metrics_by_k[k]['recall'].append(recall)
                
                # NDCG@k
                dcg = 0
                idcg = sum([1/np.log2(i+2) for i in range(min(len(relevant_products), k))])
                
                for i, pid in enumerate(retrieved_products):
                    if pid in relevant_products:
                        dcg += 1 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics_by_k[k]['ndcg'].append(ndcg)
                
                # MRR@k
                mrr = 0
                for i, pid in enumerate(retrieved_products):
                    if pid in relevant_products:
                        mrr = 1 / (i + 1)
                        break
                metrics_by_k[k]['mrr'].append(mrr)
        
        # Calculate averages
        avg_metrics = {}
        detailed_metrics = {}
        
        for k in k_values:
            avg_metrics[k] = {}
            detailed_metrics[k] = metrics_by_k[k]
            
            for metric in ['precision', 'recall', 'ndcg', 'mrr']:
                if metrics_by_k[k][metric]:
                    avg_metrics[k][metric] = np.mean(metrics_by_k[k][metric])
                else:
                    avg_metrics[k][metric] = 0.0
        
        return avg_metrics, detailed_metrics
