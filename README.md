### Phase 1: Initial Implementation (Current)
- Basic retrieval models (TF-IDF)
- Dense embeddings with pre-trained models
- Re-ranking pipeline for improved precision

### Future Work: Phase 2
- Two-tower neural architecture implementation
- Domain-specific model training
- Performance optimization and scaling

## Dataset: Amazon ESCI

Working with Amazon's ESCI (Exact, Substitute, Complement, Irrelevant) dataset:
- **Scope**: US locale product listings with exact match labels
- **Content**: Rich product metadata (titles, descriptions, attributes)
- **Sampling**: 
  - Curated 500-row evaluation set
  - 50 diverse queries for comprehensive testing
  - Stratified sampling to maintain data distribution

## Implemented Methods

## Phase 1 Implementation

### 1. Baseline: TF-IDF Search
Simple yet effective keyword-based search:
```
Features:
- Vocabulary size: 5000 terms
- Stop words removed
- Cosine similarity scoring

Performance (test query: "carpenter bench press")
✓ Found 6/8 relevant products
✓ Precision@10: 0.6000
✓ Recall@10: 0.7500
```

### 2. Dense Retrieval
Semantic search using SentenceTransformers:
```
Model: all-MiniLM-L6-v2
Storage: ChromaDB (vector database)
Features:
- 384D dense embeddings
- L2 distance metric
- Batch processing support

Performance (same test query)
✓ Found 7/8 relevant products
✓ Precision@10: 0.7000
✓ Recall@10: 0.8750
```

### 3. Re-Ranking Pipeline
Two-stage ranking for improved precision:
```
First Stage: Dense Retrieval (above)
Second Stage: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
Process:
1. Retrieve top 50 candidates
2. Re-rank with cross-encoder
3. Return top 10 results

Performance Improvement
✓ MRR increased by ~15%
✓ NDCG@10 improved
✓ Better handling of nuanced queries
```

## Technical Implementation

### Implementation Stack
```
Core Processing:
- pandas, numpy: Data handling
- scikit-learn: TF-IDF implementation
- chromadb: Vector storage

Models & Embeddings:
- sentence-transformers: Dense embeddings
- transformers: Cross-encoder model
- PyTorch: Deep learning backend
```

### Evaluation Framework
```
Metrics:
- Mean Reciprocal Rank (MRR)
- Precision@k (k=1,5,10)
- NDCG@k
- Recall@k

Process:
1. Split test queries
2. Generate rankings
3. Compare with ground truth
4. Calculate metrics
```

## Current Results Analysis

### Method Comparison
```
Approach     │ Results                    │ Pros/Cons
─────────────┼───────────────────────────┼──────────────────────────────
TF-IDF       │ • MRR: 0.4521            │ + Fast, simple deployment
             │ • Precision@10: 0.6000    │ + Memory efficient
             │ • Recall@10: 0.7500      │ - Word-based matching only
             │ • HITS@1: 0.3245         │ - Misses semantic relationships
─────────────┼───────────────────────────┼──────────────────────────────
Dense Search │ • MRR: 0.5632            │ + Better semantic understanding
             │ • Precision@10: 0.7000    │ + Handles synonyms well
             │ • Recall@10: 0.8750      │ - Higher computational cost
             │ • HITS@1: 0.4521         │ - Requires vector storage
─────────────┼───────────────────────────┼──────────────────────────────
Re-Ranking   │ • MRR: 0.6473            │ + Highest precision
             │ • Precision@10: 0.7500    │ + Best ranking quality
             │ • Recall@10: 0.8750      │ - Increased latency
             │ • HITS@1: 0.5234         │ - Complex two-stage pipeline
```
*Results based on evaluation across all test queries

## Setup & Usage

### Installation
```bash
# Core dependencies
pip install pandas numpy scikit-learn
pip install sentence-transformers transformers
pip install chromadb tqdm

# Optional for visualization
pip install matplotlib seaborn
```

### Project Structure
```
.
├── final.ipynb              # Main implementation notebook
├── sample_2c_full_data.csv  # Evaluation dataset
├── requirements.txt         # Dependencies
└── two_tower_build/        # Future implementation
```

## Future Work (Phase 2)

### Two-Tower Neural Architecture
```
Planned Features:
- Custom query/product encoders
- Domain-specific training
- Efficient indexing

Goals:
1. Better semantic matching
2. Faster retrieval
3. Domain adaptation
```

### Implementation Steps
1. Data preparation pipeline
2. Model architecture design
3. Training infrastructure setup
4. Evaluation framework
5. Production deployment plan

### Expected Benefits
- Improved relevance scores
- Lower latency at scale
- Better handling of domain-specific queries
- Easier maintenance and updates

## Two-Tower Implementation Details

### Architecture Overview

The two-tower model implementation follows these key design principles:

1. **Dual Encoder Architecture**
   - Query encoder tower
   - Product encoder tower
   - Shared embedding space for efficient similarity computation

2. **Training Approach**
   - Triplet loss optimization
   - Hard negative mining
   - Domain-specific fine-tuning

3. **Product Indexing**
   - Efficient vector storage
   - Fast approximate nearest neighbor search
   - Batch processing for scalability

### Build Process

The build system is organized in the `two_tower_build/` directory:

1. **Model Training (`two_tower.py`)**
   - Data preprocessing and batching
   - Model architecture definition
   - Training loop implementation
   - Checkpoint management

2. **Evaluation Framework (`two_tower_evaluation.py`)**
   - Metrics calculation (Precision, Recall, NDCG, MRR)
   - Product index construction
   - Query processing pipeline
   - Results analysis and visualization

3. **Model Artifacts**
   - Saved in `two_tower_model/` directory
   - Includes tokenizer configuration
   - Contains model weights and parameters
   - Serialized vocabulary files