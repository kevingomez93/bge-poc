# BGE Proof-of-Concept

This repository contains a proof-of-concept project demonstrating the use of the `FlagEmbedding` library to generate text embeddings and perform similarity searches.

## Overview

The `bge_poc.py` script showcases the core functionalities of the BGE (BAAI General Embedding) models. It initializes a pre-trained model, encodes sentences into dense vectors, and computes their similarity. Additionally, it includes a simple retrieval example where a query is used to find the most relevant document in a small corpus.

## Features

- **Text Embedding**: Converts sentences into high-dimensional vectors.
- **Similarity Calculation**: Computes the cosine similarity between sentence embeddings.
- **Information Retrieval**: Demonstrates a basic search functionality by matching a query to a corpus.

## Getting Started

### Prerequisites

- Python 3
- Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kevingomez93/bge-poc.git
    cd bge-poc
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Script

To run the proof-of-concept, execute the following command:

```bash
python bge_poc.py
```

The script will download the pre-trained model, perform the embedding and similarity calculations, and print the results to the console. 




<img width="1330" alt="image" src="https://github.com/user-attachments/assets/d1641dbb-ccae-498a-a048-c90fea9f5d06" />
