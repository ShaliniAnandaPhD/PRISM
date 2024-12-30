import json
import numpy as np
from typing import List, Dict
from pydantic import BaseModel

class BayesianConfig(BaseModel):
    """
    Configuration for Bayesian Model Averaging.
    """
    evaluation_results_path: str  # Path to the evaluation results JSON file
    model_priors: Dict[str, float]  # Prior probabilities for each model
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

def bayesian_model_averaging(responses: List[str], scores: List[float], priors: List[float]) -> str:
    """
    Applies Bayesian Model Averaging to determine the final output.

    Args:
        responses (List[str]): Responses from different models.
        scores (List[float]): Relevance scores assigned to each response.
        priors (List[float]): Prior probabilities for each model.

    Returns:
        str: Final consensus response.
    """
    weighted_scores = np.array(scores) * np.array(priors)
    posterior_probabilities = weighted_scores / np.sum(weighted_scores)
    selected_index = np.argmax(posterior_probabilities)
    return responses[selected_index]

def process_bayesian_averaging(config: BayesianConfig):
    """
    Processes evaluation results and applies Bayesian Model Averaging to aggregate model outputs.

    Args:
        config (BayesianConfig): Configuration containing paths and priors.
    """
    # Load evaluation results
    results = load_evaluation_results(config.evaluation_results_path)

    # Initialize aggregated results
    aggregated_results = []

    # Process each prompt and its responses
    for i, prompt in enumerate(results["prompts"]):
        responses = [results["openai"][i], results["mistral"][i], results["claude"][i]]
        scores = [results["openai_scores"][i], results["mistral_scores"][i], results["claude_scores"][i]]
        priors = [config.model_priors["openai"], config.model_priors["mistral"], config.model_priors["claude"]]

        # Perform Bayesian averaging
        final_response = bayesian_model_averaging(responses, scores, priors)

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
            "priors": priors,
        })

    # Save aggregated results
    with open(config.output_path, "w") as file:
        json.dump(aggregated_results, file, indent=4)
    print(f"Aggregated results saved to {config.output_path}")

if __name__ == "__main__":
    """
    Entry point for Bayesian Model Averaging script.

    What We Did:
    - Loaded evaluation results and priors:
      - Combined responses, scores, and priors to compute posterior probabilities.

    - Bayesian Averaging:
      - Used prior probabilities and model scores to compute weighted consensus outputs.
      - Selected the response with the highest posterior probability.

    - Saved Aggregated Outputs:
      - Stored final consensus results alongside raw responses, scores, and priors in `aggregated_results.json`.

    What's Next:
    - Explainable Analysis:
      - Visualize the impact of priors and scores on final consensus using Explainable AI tools.

    - Integration with Referee LLM:
      - Resolve ambiguous or low-confidence cases using a specialized referee model.

    - Build RAG Pipeline:
      - Integrate consensus outputs into a Retrieval-Augmented Generation workflow.
    """
    # Define the configuration
    config = BayesianConfig(
        evaluation_results_path="./model_evaluation_results.json",
        model_priors={
            "openai": 0.5,
            "mistral": 0.3,
            "claude": 0.2,
        },
        output_path="./aggregated_results_bayesian.json",
    )

    # Run the Bayesian averaging process
    process_bayesian_averaging(config)
