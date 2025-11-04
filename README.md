## Interesting findings from the sample

While curating and exploring the 500-row sample, a few notable and sometimes surprising items appeared:

- Exact product code: `usb2aub2ra1m` — this looked like a query but appears to be a specific product identifier (a right-angled USB connector in the sample).
- Unusual query: "fake urine" (humorous / unexpected user intent surfaced in queries).
- Plumbing-related query example: "zurn qkipsp 5 port plastic manifold without valves" — shows highly specific technical queries that benefit from strong term/attribute matching.

These examples show why we use a mixed approach in Phase 1 (TF-IDF + dense retrieval + re-ranking): some queries need exact-token matching (product codes, part numbers), while others benefit from semantic understanding and re-ranking to surface the best match.

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
