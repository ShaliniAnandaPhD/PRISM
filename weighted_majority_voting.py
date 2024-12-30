import json
import numpy as np
from typing import List, Dict
from pydantic import BaseModel

class VotingConfig(BaseModel):
    """
    Configuration for the Weighted Majority Voting process.
    """
    evaluation_results_path: str  # Path to the evaluation results JSON file
    model_weights: Dict[str, float]  # Weights assigned to each model
    output_path: str  # Path to save the final aggregated results

def load_evaluation_results(file_path: str) -> Dict[str, List[str]]:
    """
    Loads evaluation results from a JSON file.

    Args:
        file_path (str): Path to the evaluation results JSON file.

    Returns:
        Dict[str, List[str]]: Dictionary containing evaluation results for each model.
    """
    with open(file_path, "r") as file:
        results = json.load(file)
    print(f"Loaded evaluation results from {file_path}")
    return results

def weighted_majority_vote(responses: List[str], scores: List[float], weights: List[float]) -> str:
    """
    Applies Weighted Majority Voting to determine the final output.

    Args:
        responses (List[str]): Responses from different models.
        scores (List[float]): Relevance scores assigned to each response.
        weights (List[float]): Weights assigned to each model.

    Returns:
        str: Final consensus response.
    """
    weighted_scores = np.array(scores) * np.array(weights)
    selected_index = np.argmax(weighted_scores)
    return responses[selected_index]

def process_voting(config: VotingConfig):
    """
    Processes evaluation results and applies Weighted Majority Voting to aggregate model outputs.

    Args:
        config (VotingConfig): Configuration containing paths and weights.
    """
    # Load evaluation results
    results = load_evaluation_results(config.evaluation_results_path)

    # Initialize aggregated results
    aggregated_results = []

    # Process each prompt and its responses
    for i, prompt in enumerate(results["prompts"]):
        responses = [results["openai"][i], results["mistral"][i], results["claude"][i]]
        scores = [results["openai_scores"][i], results["mistral_scores"][i], results["claude_scores"][i]]
        weights = [config.model_weights["openai"], config.model_weights["mistral"], config.model_weights["claude"]]

        # Perform weighted voting
        final_response = weighted_majority_vote(responses, scores, weights)

        # Collect results
        aggregated_results.append({
            "prompt": prompt,
            "final_response": final_response,
            "responses": {
                "openai": responses[0],
                "mistral": responses[1],
                "claude": responses[2],
            },
            "scores": {
                "openai": scores[0],
                "mistral": scores[1],
                "claude": scores[2],
            },
            "weights": weights,
        })

    # Save aggregated results
    with open(config.output_path, "w") as file:
        json.dump(aggregated_results, file, indent=4)
    print(f"Aggregated results saved to {config.output_path}")

if __name__ == "__main__":
    """
    Entry point for the Weighted Majority Voting script.
    
    What We Did:
    - Integrated Evaluation Results:
      - Loaded responses and relevance scores from the `model_evaluation_results.json` file.

    - Weighted Voting Mechanism:
      - Used predefined weights and relevance scores to calculate the most reliable response.
      - Selected the response with the highest weighted score.

    - Saved Aggregated Outputs:
      - Stored the final consensus results alongside raw responses and scores in `aggregated_results.json`.

    What's Next:
    - Bayesian Model Averaging:
      - Extend the aggregation process with probabilistic techniques to combine insights across models.

    - Explainable Analysis:
      - Visualize how model weights and scores influence the consensus outputs using Explainable AI techniques.

    - Integrate with RAG Pipeline:
      - Combine these aggregated results into a retrieval-augmented generation pipeline for downstream applications.
    """
    # Define the configuration
    config = VotingConfig(
        evaluation_results_path="./model_evaluation_results.json",
        model_weights={
            "openai": 0.4,
            "mistral": 0.35,
            "claude": 0.25,
        },
        output_path="./aggregated_results.json",
    )

    # Run the voting process
    process_voting(config)
