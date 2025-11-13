# evaluate.py
import pandas as pd
from tqdm import tqdm
import warnings
from dotenv import load_dotenv
import time

# Suppress all warnings for cleaner output
warnings.filterwarnings("ignore")

load_dotenv()

# Import components 
from app import setup_agent 
from evaluation_data import EVAL_DATASET
from metrics import calculate_f1_score, calculate_semantic_similarity

def run_evaluation():
    """
    Orchestrates the RAG agent evaluation process from start to finish.
    """
    print("Setting up the RAG agent for evaluation...")
    agent_executor = setup_agent()
    print("✅ Agent setup complete.")

    results = []
    
    # Loop through questions 
    for item in tqdm(EVAL_DATASET, desc="Evaluating Agent Performance"):
        question = item["question"]
        ground_truth = item["ground_truth_answer"]

        try:    
            # Get the agent's response
            response = agent_executor.invoke({
                "input": question,
                "chat_history": []
            })
            generated_answer = response.get("output", "Error: No output from agent.")

            # Calculate both scores
            f1_scores = calculate_f1_score(generated_answer, ground_truth)
            similarity_score = calculate_semantic_similarity(generated_answer, ground_truth)

            results.append({
                "question": question,
                "f1_score": f1_scores["f1_score"],
                "semantic_similarity": similarity_score,
                "precision": f1_scores["precision"],
                "recall": f1_scores["recall"],
                "generated_answer": generated_answer,
            })
        except Exception as e:
            print(f"\nAn error occurred on question: '{question}'. Error: {e}")
            # Optionally add a placeholder result so the script can continue
            results.append({
                "question": question,
                "f1_score": 0,
                "semantic_similarity": 0,
                "generated_answer": f"API Error: {e}",
            })
        
        # Add a 5-second delay to stay within the free tier rate limit
        time.sleep(5) 

    # Create a pandas DataFrame for a clean summary report
    df = pd.DataFrame(results)
    
    # Calculate and print the average scores for the whole dataset
    avg_semantic_similarity = df['semantic_similarity'].mean()
    avg_f1_score = df['f1_score'].mean()
    avg_precision = df['precision'].mean()
    avg_recall = df['recall'].mean()

    print("\n--- 📊 Evaluation Report ---")
    print(df[['question', 'f1_score','semantic_similarity']].to_string())
    print("\n--- 📈 Average Scores ---")
    print(f"Average Semantic Similarity: {avg_semantic_similarity:.4f}")
    print(f"Average F1 Score: {avg_f1_score:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    

    # Save the detailed results to a CSV file for analysis
    df.to_csv("rag_evaluation_results.csv", index=False)
    print("\n📄 Full results saved to rag_evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()