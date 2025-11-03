## Overview

This project implements and compares multiple approaches to e-commerce product search, addressing the challenge of matching user queries to relevant products. Using Amazon's ESCI dataset, we evaluate traditional information retrieval methods alongside modern deep learning approaches.

## Dataset

The project utilizes Amazon's ESCI (Exact, Substitute, Complement, Irrelevant) dataset:
- **181,819 query-product pairs** from real e-commerce interactions
- **Product relationships** categorized as Exact matches, Substitutes, Complements, or Irrelevant
- **Sample dataset**: 485 rows across 50 unique queries for focused experimentation

## Implemented Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- Traditional keyword-based matching with statistical weighting
- Fast and interpretable baseline approach
- Performance: 6/8 relevant products found for test query "carpenter bench press"

### 2. Dense Embeddings
- Semantic understanding using `all-MiniLM-L6-v2` sentence transformer
- Captures meaning beyond exact keyword matches
- Performance: 7/8 relevant products found for the same test query

### 3. Two-Tower Neural Network
- Custom PyTorch implementation with separate encoders for queries and products
- Trained using triplet loss for domain-specific learning
- Designed for e-commerce query-product matching optimization

## Technical Stack

- **pandas & numpy**: Data processing and manipulation
- **scikit-learn**: Traditional ML algorithms and metrics
- **sentence-transformers**: Pre-trained semantic embedding models
- **PyTorch & transformers**: Deep learning framework and models
- **ChromaDB**: Vector database for efficient similarity search
- **matplotlib & seaborn**: Data visualization

## Model Comparison

### TF-IDF
- **Approach**: Keyword-based matching with statistical weighting
- **Pros**: Fast, interpretable, works well for exact matches
- **Cons**: Limited semantic understanding

### Dense Embeddings
- **Approach**: Semantic similarity using pre-trained transformers
- **Pros**: Understands synonyms, context, and semantic relationships
- **Cons**: May over-generalize for specific domain queries

### Two-Tower Model
- **Approach**: Domain-specific neural architecture with separate query and product encoders
- **Pros**: Optimized for e-commerce, learns complex query-product relationships
- **Cons**: Requires substantial training data and computational resources

## Installation and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train the two-tower model
python two_tower_final.py

# Evaluate model performance
python two_tower_evaluation.py
```

## Project Structure

- **`two_tower_final.py`** - Two-tower model implementation and training
- **`two_tower_evaluation.py`** - Model evaluation and metrics calculation
- **`test.ipynb`** - Experimental notebook with data exploration and model comparisons
- **`sample_2c_full_data.csv`** - Curated dataset sample for experiments
- **`dataset/`** - Original Amazon ESCI dataset files

## Results

Performance comparison on test query "carpenter bench press":
- **TF-IDF**: 6/8 relevant products retrieved
- **Dense Model**: 7/8 relevant products retrieved
- **Two-Tower**: Training in progress 
