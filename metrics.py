# metrics.py
import re
from collections import Counter
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_f1_score(generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """
    Calculates token-level Precision, Recall, and F1 score between two strings.
    """
    # Normalize and tokenize text into a frequency map of words
    def tokenize(text: str) -> Counter:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return Counter(tokens)

    # Get token counters for both the generated and ground truth answers
    generated_tokens = tokenize(generated_answer)
    ground_truth_tokens = tokenize(ground_truth_answer)

    # Calculate True Positives (TP): common tokens between both answers
    common_tokens = generated_tokens & ground_truth_tokens
    tp = sum(common_tokens.values())

    # Calculate False Positives (FP): tokens in generated answer but not in ground truth
    fp = sum(generated_tokens.values()) - tp

    # Calculate False Negatives (FN): tokens in ground truth but not in generated answer
    fn = sum(ground_truth_tokens.values()) - tp

    # Calculate precision, recall, and F1 score, handling division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_semantic_similarity(generated_answer: str, ground_truth_answer: str) -> float:
    """
    Calculates the semantic similarity between two sentences using a transformer model.
    Returns a score between -1 and 1 (typically 0 to 1).
    """
    # Encode the sentences into vectors
    embeddings = embedding_model.encode([generated_answer, ground_truth_answer])
    
    # Calculate cosine similarity. The function expects 2D arrays, so we reshape.
    # embeddings[0].reshape(1, -1) gets the vector for the first sentence.
    similarity_score = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0] # The result is a 2D array, so we extract the single value
    
    return float(similarity_score)

# This block allows to test this file directly
if __name__ == "__main__":
    truth = "BERT is pre-trained on Masked Language Modeling and Next Sentence Prediction."
    generated = "BERT is pre-trained using Masked Language Modeling."
    scores = calculate_f1_score(generated, truth)
    print(f"Example Scores: {scores}")