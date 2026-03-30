# TBI TP 2
## Search Engine from Scratch

This project implements a basic search engine built from scratch in Python. It implements some basic indexing and retrieval methods and provides some more advanced methods as well.

## Features

- **Improved Preprocessing Pipeline**: Uses a better tokenizer, stemming, and stop words removal.
- **Indexing Algorithms**: 
  - **BSBI (Block Sort-Based Indexing)**: Indexes predefined blocks of the document collection in memory and merges them continuously to support large collections. Requires a global term-termID map to be shared across all blocks and stores mapping between termIDs and posting lists.
  - **SPIMI (Single-Pass In-Memory Indexing)**: Fast single-pass indexing that continuously flushes inverted lists onto disk blocks without requiring a global term-termID map. Each block maintains its own dictionary of terms (not termIDs) to posting lists, thus reducing overhead and improving sorting speed.
- **Latent Semantic Indexing (LSI)**: Includes an LSI implementation using Truncated SVD for fast SVD calculation on large, sparse matrices alongside `FAISS` for approximate nearest neighbor rapid, dense vector-based similarity retrieval that maps terms to semantic concepts.
- **Postings Compression**:
  - Provides 4 varying algorithms to minimize memory & disk usages for postings lists representations: 
    - Uncompressed postings (`StandardPostings`): Stores the posting list uncompressed for fast encoding and decoding, but requires more disk space and memory.
    - Variable Byte Encoding (`VBEPostings`): Compresses the posting list using delta encoding (gap-based postings list), then uses Variable Byte Encoding, a byte-level compression scheme, to store the gaps using a variable number of bytes.
    - OptPForDelta (`OptPForDeltaPostings`): Compresses the posting list using delta encoding (gap-based postings list), then uses an optimized version of PForDelta, a bit-level compression scheme, to pack a block of integers with b bits per integer. The original PForDelta algorithm aims to choose b such that only 10% of integers will require more than b bits (exceptions). The optimized version instead chooses b such that it achieves the best compression size possible by calculating the space required for storing both regular integers and exceptions. For simplicity, exceptions are stored by their indices and raw values as plain 32 bit integers.
    - Fast Binary Packing (`BP128Postings`): Compresses the posting list using delta encoding (gap-based postings list), then packs blocks of 128 integers using b bits, where b is the number of bits required for the largest integer in the block. Provides fast encoding and decoding.
- **Compressed Dictionary Terms**:
  - The Term-Postings list dictionary utilizes an efficient **Finite State Transducer (FST)** representation backed by the backend Rust implementation (`rust-fst`). This enables compact dictionary sizes and fast disk-read retrieval without inflating main memory usage.
- **Information Retrieval Scoring Frameworks**:
  - **TF-IDF Model**: 
  - **BM25 Probabilistic Model**
  - **Optimized BM25-WAND (Weak AND)**: The maximum BM25 score per term is precomputed during merge-time to implement the WAND algorithm. This allows efficient pruning for document evaluations on top-k retrieval, improving latency heavily on large collections.
- **Evaluation Functions**:
  - Generates metrics for various retrieval schemes such as Rank Biased Precision (RBP), (Normalized) Discounted Cumulative Gain (DCG/NDCG), and Mean Average Precision (MAP).

## Installation
1. Create a virtual environment and activate it (optional)
```bash
python -m venv env  # Creates a virtual env in the `env` directory
env\Scripts\Activate
```
2. Install required libraries using `pip`
```bash
pip install -r requirements.txt
```

## How to Run
### 1. Generating the Index

Before performing retrieval, run the relevant indexing scripts to generate index files depending on the desired retrieval scheme is desired.

#### Using BSBI
```bash
python bsbi.py --compression [COMPRESSION_ALGORITHM] [--compare]
```
**Options:**
- `--compression [COMPRESSION_ALGORITHM]`: Determines the compression algorithm used to compress the postings lists. `COMPRESSION_ALGORITHM` must be one of `standard`, `vbe`, `optpfor`, or `bp128`. If not specified, defaults to `vbe`.
- `--compare`: If specified, ignores `--compression` and runs all compression schemes, then prints the final index sizes for each scheme. Useful to compare the performance of different compression algorithms.

**Example:**
```bash
python bsbi.py --compression bp128
```

#### Using SPIMI
```bash
python spimi.py --compression [COMPRESSION_ALGORITHM] [--compare] --max-docs [MAX_DOCS]
```
**Options:**
- `--compression [COMPRESSION_ALGORITHM]`: Determines the compression algorithm used to compress the postings lists. `COMPRESSION_ALGORITHM` must be one of `standard`, `vbe`, `optpfor`, or `bp128`. If not specified, defaults to `vbe`.
- `--compare`: If specified, ignores `--compression` and runs all compression schemes, then prints the final index sizes for each scheme. Useful to compare the performance of different compression algorithms.
- `--max-docs [MAX_DOCS]`: Limits the number of documents to index in a single iteration. Used to control memory usage and the number of blocks generated. If not specified, defaults to `100`.

**Example:**
```bash
python spimi.py --compare --max-docs 50
```

#### Using LSI
```bash
python lsi.py --k [K]
```
**Options:**
- `--k [K]`: The number of latent dimensions to reduce the term-document matrix to. If not specified, defaults to `100`.

**Example:**
```bash
python lsi.py --k 200
```

### 2. Searching & Retrieving
Search over the built index by passing a specific search query, or by running predefined queries.

```bash
python search.py --scoring [SCORING_MODEL] --compression [COMPRESSION_ALGORITHM] --query [QUERY_TEXT] [--spimi] [--verbose]
```

**Options:**
- `--scoring [SCORING_MODEL]`: Metric algorithm used to evaluate top document relevance. `SCORING_MODEL` must be one of `tf-idf`, `bm25`, `bm25-wand`, or `lsi`. If unspecified, defaults to `tf-idf`.
- `--query [QUERY_TEXT]`: Freeform text to query over the index. If unspecified, runs over predefined sample queries.
- `--verbose`: Includes runtime duration and evaluated document counts for each query.
- `--spimi`: Use this flag if the SPIMI indexing scheme was used during indexing.
- `--compression [COMPRESSION_ALGORITHM]`: Manages the algorithm used to decompress the postings lists. `COMPRESSION_ALGORITHM` must be one of `standard`, `vbe`, `optpfor`, or `bp128`. If not specified, defaults to `vbe`. Ensure this parameter is configured to match the compression algorithm used during indexing.

**Example:**
```bash
python search.py --scoring bm25-wand --compression optpfor --query "example search query" --spimi --verbose
```

### 3. Execution Effectiveness & Evaluation
Compute retrieval results using standardized evaluation metrics. Runs queries defined in `queries.txt` and references them with the corresponding relevance judgments in `qrels.txt`.

```bash
python evaluation.py --eval [EVALUATION_METRIC] --scoring [SCORING_MODEL] --compression [COMPRESSION_ALGORITHM] [--spimi]
```
**Options:**
- `--eval [EVALUATION_METRIC]`: The evaluation metric to compute. `EVALUATION_METRIC` must be one of `rbp`, `dcg`, `ndcg`, or `ap`. If not specified, defaults to `rbp`.
- `--scoring [SCORING_MODEL]`: The scoring model to be used during retrieval. `SCORING_MODEL` must be one of `tf-idf`, `bm25`, `bm25-wand`, or `lsi`. If not specified, defaults to `tf-idf`.
- `--compression [COMPRESSION_ALGORITHM]`: Manages the algorithm used to decompress the postings lists. `COMPRESSION_ALGORITHM` must be one of `standard`, `vbe`, `optpfor`, or `bp128`. If not specified, defaults to `vbe`. Ensure this parameter is configured to match the compression algorithm used during indexing.
- `--spimi`: Use this flag if the SPIMI indexing scheme was used during indexing.

**Example:**
```bash
python evaluation.py --eval ndcg --scoring bm25-wand --compression optpfor --spimi
```