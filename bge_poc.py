import numpy as np
from FlagEmbedding import FlagModel

def main():
    sentences_1 = ["What is BGE on HuggingFace?", "The M3E mission is to train an massive multi-lingual embedding model."]
    sentences_2 = ["fastapi is a modern, fast, web framework for building APIs with Python", "FlagEmbedding can map any text to a low-dimensional dense vector"]

    model = FlagModel('BAAI/bge-large-en-v1.5', 
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                      use_fp16=False)

    print("Encoding sentences...")
    embeddings_1 = model.encode(sentences_1)
    embeddings_2 = model.encode(sentences_2)
    
    similarity = embeddings_1 @ embeddings_2.T
    print("Similarity matrix:")
    print(similarity)

    # A simple retrieval example
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The capital of France is Paris.",
        "Artificial intelligence is a branch of computer science.",
        "The Great Wall of China is a series of fortifications.",
        "Photosynthesis is a process used by plants to convert light energy into chemical energy."
    ]
    query = "What is the capital of France?"

    print("\n--- Simple Retrieval Example ---")
    print(f"Query: {query}")

    query_embedding = model.encode([query])
    corpus_embeddings = model.encode(corpus)

    # Calculate cosine similarity
    similarities = (query_embedding @ corpus_embeddings.T)[0]
    
    # Find the most similar sentence
    most_similar_idx = np.argmax(similarities)
    
    print(f"\nCorpus:")
    for i, sentence in enumerate(corpus):
        print(f"  - {sentence} (Similarity: {similarities[i]:.4f})")
    
    print(f"\nMost similar sentence in corpus:")
    print(f"  - '{corpus[most_similar_idx]}' with a similarity score of {similarities[most_similar_idx]:.4f}")

if __name__ == "__main__":
    main() 