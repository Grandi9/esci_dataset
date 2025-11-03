import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import warnings
import random
import os

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower model training"""
    
    def __init__(self, queries, pos_products, neg_products, tokenizer, max_length=128):
        self.queries = queries
        self.pos_products = pos_products
        self.neg_products = neg_products
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = str(self.queries[idx])
        pos_product = str(self.pos_products[idx])
        neg_product = str(self.neg_products[idx])
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize positive product
        pos_encoding = self.tokenizer(
            pos_product,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize negative product
        neg_encoding = self.tokenizer(
            neg_product,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(),
            'pos_input_ids': pos_encoding['input_ids'].squeeze(),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(),
        }

class TwoTowerModel(nn.Module):
    """Two-Tower Architecture for Query-Product Matching"""
    
    def __init__(self, model_name='distilbert-base-uncased', embedding_dim=768):
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
    
    def forward(self, query_input_ids, query_attention_mask, 
                pos_input_ids, pos_attention_mask,
                neg_input_ids, neg_attention_mask):
        
        # Encode query
        query_emb = self.encode_query(query_input_ids, query_attention_mask)
        
        # Encode positive product
        pos_emb = self.encode_product(pos_input_ids, pos_attention_mask)
        
        # Encode negative product
        neg_emb = self.encode_product(neg_input_ids, neg_attention_mask)
        
        return query_emb, pos_emb, neg_emb

def triplet_loss_corrected(query_emb, pos_emb, neg_emb, margin=0.2):
    """CORRECTED Triplet loss function"""
    # Calculate similarities (cosine similarity since embeddings are normalized)
    pos_similarity = torch.sum(query_emb * pos_emb, dim=1)
    neg_similarity = torch.sum(query_emb * neg_emb, dim=1)
    
    # CORRECT: We want pos_similarity > neg_similarity + margin
    # Loss = max(0, margin + neg_similarity - pos_similarity)
    loss = torch.clamp(margin + neg_similarity - pos_similarity, min=0.0)
    return torch.mean(loss)

def validate_model(model, val_queries, val_pos_products, val_neg_products, tokenizer, device):
    """Validate model during training"""
    model.eval()
    total_loss = 0
    correct_rankings = 0
    total_samples = min(100, len(val_queries))  # Sample for validation
    
    with torch.no_grad():
        for i in range(total_samples):
            # Tokenize
            query_enc = tokenizer(val_queries[i], truncation=True, padding='max_length', 
                                max_length=128, return_tensors='pt')
            pos_enc = tokenizer(val_pos_products[i], truncation=True, padding='max_length', 
                              max_length=128, return_tensors='pt')
            neg_enc = tokenizer(val_neg_products[i], truncation=True, padding='max_length', 
                              max_length=128, return_tensors='pt')
            
            # Move to device
            query_ids = query_enc['input_ids'].to(device)
            query_mask = query_enc['attention_mask'].to(device)
            pos_ids = pos_enc['input_ids'].to(device)
            pos_mask = pos_enc['attention_mask'].to(device)
            neg_ids = neg_enc['input_ids'].to(device)
            neg_mask = neg_enc['attention_mask'].to(device)
            
            # Forward pass
            query_emb = model.encode_query(query_ids, query_mask)
            pos_emb = model.encode_product(pos_ids, pos_mask)
            neg_emb = model.encode_product(neg_ids, neg_mask)
            
            # Calculate loss
            loss = triplet_loss_corrected(query_emb, pos_emb, neg_emb)
            total_loss += loss.item()
            
            # Check if positive is ranked higher than negative
            pos_sim = torch.sum(query_emb * pos_emb, dim=1)
            neg_sim = torch.sum(query_emb * neg_emb, dim=1)
            if pos_sim > neg_sim:
                correct_rankings += 1
    
    model.train()
    avg_loss = total_loss / total_samples
    ranking_accuracy = correct_rankings / total_samples
    
    return avg_loss, ranking_accuracy

def train_two_tower_model_corrected():
    print("--- CORRECTED Two-Tower Model Training ---")
    
    # Load and prepare data
    try:
        df_examples = pd.read_parquet('dataset/shopping_queries_dataset_examples.parquet')
        df_products = pd.read_parquet('dataset/shopping_queries_dataset_products.parquet')
        df_sample_2c = pd.read_csv('sample_2c_full_data.csv')
        print("Successfully loaded all datasets.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    print(f"Loaded {len(df_examples)} examples and {len(df_products)} products.")
    
    # Merge datasets
    df_merged = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale', 'product_id'],
        right_on=['product_locale', 'product_id']
    )
    
    # Filter for Task 1
    df_task_1 = df_merged[df_merged["small_version"] == 1].copy()
    
    # Get test queries to exclude (CRITICAL!)
    test_queries = set(df_sample_2c['query'].unique())
    print(f"🚨 EXCLUDING {len(test_queries)} test queries from training to prevent data leakage")
    
    # Create IMPROVED product text mapping (prioritize title and description)
    product_cols = ['product_title', 'product_brand', 'product_color', 'product_description', 'product_bullet_point']
    for col in product_cols:
        df_merged[col] = df_merged[col].fillna('')
    
    # IMPROVED: Prioritize title and description, limit length
    df_merged['product_text'] = (
        df_merged['product_title'] + ' ' +
        df_merged['product_description'].str[:200] + ' ' +  # Limit description
        df_merged['product_brand'] + ' ' +
        df_merged['product_bullet_point'].str[:100]  # Limit bullet points
    )
    df_merged['product_text'] = df_merged['product_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Limit overall product text length to avoid noise
    df_merged['product_text'] = df_merged['product_text'].str[:400]
    
    product_map = dict(zip(df_merged['product_id'], df_merged['product_text']))
    
    # Build training set (EXCLUDE test queries!)
    df_train_source = df_task_1[df_task_1['product_locale'] == 'us'].copy()
    df_train_source = df_train_source[~df_train_source['query'].isin(test_queries)]
    
    print(f"Training data: {len(df_train_source)} rows (after excluding test queries)")
    
    # Get positive and negative pairs
    positives = df_train_source[df_train_source['esci_label'] == 'E'][['query', 'product_id']]
    negatives = df_train_source[df_train_source['esci_label'] == 'I'][['query', 'product_id']]
    
    # Create triplets
    df_triplets = pd.merge(
        positives,
        negatives,
        on='query',
        suffixes=('_pos', '_neg')
    )
    
    print(f"Found {len(df_triplets)} potential triplets.")
    
    # DIAGNOSTIC: Check training data quality
    print("\n🔍 TRAINING DATA DIAGNOSTICS:")
    print(f"Positive examples (E): {len(positives)}")
    print(f"Negative examples (I): {len(negatives)}")
    
    # Sample some triplets for inspection
    if len(df_triplets) > 0:
        sample_triplet = df_triplets.iloc[0]
        sample_query = sample_triplet['query']
        sample_pos_id = sample_triplet['product_id_pos']
        sample_neg_id = sample_triplet['product_id_neg']
        
        print(f"\n📝 SAMPLE TRIPLET:")
        print(f"Query: '{sample_query}'")
        print(f"Positive Product: '{product_map.get(sample_pos_id, 'N/A')[:100]}...'")
        print(f"Negative Product: '{product_map.get(sample_neg_id, 'N/A')[:100]}...'")
    
    if len(df_triplets) == 0:
        print("❌ No triplets found. Exiting.")
        return
    
    # Sample triplets - INCREASE for better training
    N_TRAINING_EXAMPLES = min(7500, len(df_triplets))
    df_triplets_sample = df_triplets.sample(N_TRAINING_EXAMPLES, random_state=42)
    
    print(f"\n📊 Using {N_TRAINING_EXAMPLES} training examples")
    
    # Prepare training data
    queries = []
    pos_products = []
    neg_products = []
    
    for _, row in tqdm(df_triplets_sample.iterrows(), total=len(df_triplets_sample), desc="Preparing triplets"):
        query = row['query']
        pos_text = product_map.get(row['product_id_pos'], '')
        neg_text = product_map.get(row['product_id_neg'], '')
        
        if query and pos_text and neg_text:
            queries.append(query)
            pos_products.append(pos_text)
            neg_products.append(neg_text)
    
    print(f"Prepared {len(queries)} training triplets.")
    
    # Split into train/val
    split_idx = int(0.9 * len(queries))
    train_queries = queries[:split_idx]
    train_pos = pos_products[:split_idx]
    train_neg = neg_products[:split_idx]
    
    val_queries = queries[split_idx:]
    val_pos = pos_products[split_idx:]
    val_neg = neg_products[split_idx:]
    
    print(f"Train: {len(train_queries)}, Validation: {len(val_queries)}")
    
    # Initialize model and tokenizer
    MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TwoTowerModel(model_name=MODEL_NAME, embedding_dim=256)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TwoTowerDataset(train_queries, train_pos, train_neg, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Increased batch size
    
    # Setup optimizer - ADJUSTED learning rate
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Slightly higher LR + weight decay
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    # Training loop
    model.train()
    num_epochs = 3  # MORE epochs for better learning
    
    print(f"Starting training for {num_epochs} epoch(s)...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(device)
            
            # Forward pass
            query_emb, pos_emb, neg_emb = model(
                batch['query_input_ids'], batch['query_attention_mask'],
                batch['pos_input_ids'], batch['pos_attention_mask'],
                batch['neg_input_ids'], batch['neg_attention_mask']
            )
            
            # Compute CORRECTED loss
            loss = triplet_loss_corrected(query_emb, pos_emb, neg_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        
        # Validation
        val_loss, val_accuracy = validate_model(model, val_queries, val_pos, val_neg, tokenizer, device)
        
        # Step learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, LR: {current_lr:.2e}")
        
        # Early stopping if validation accuracy is very low
        if epoch > 0 and val_accuracy < 0.1:
            print(f"⚠️  Low validation accuracy ({val_accuracy:.4f}). Consider checking training data quality.")
    
    # Save model
    output_dir = './two_tower_model_corrected'
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    tokenizer.save_pretrained(output_dir)
    
    print(f"CORRECTED model saved to {output_dir}")
    print("Training completed successfully!")

if __name__ == "__main__":
    train_two_tower_model_corrected()
