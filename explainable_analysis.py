import shap
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline
from typing import List, Dict

class ExplainableConfig(BaseModel):
    """
    Configuration for the Explainable Analysis process.
    """
    referee_results_path: str  # Path to the JSON file containing referee results
    model_name: str  # Hugging Face model name to load
    output_path: str  # Path to save the SHAP explanations

def load_referee_results(file_path: str) -> pd.DataFrame:
    """
    Load referee results into a DataFrame.

    Args:
        file_path (str): Path to the JSON file containing referee results.

    Returns:
        pd.DataFrame: DataFrame containing the prompts, responses, and explanations.
    """
    results = pd.read_json(file_path)
    print(f"Loaded referee results from {file_path}")
    return results

class ExplainableAnalysis:
    """
    Class to perform explainable analysis on model decisions using SHAP.
    """

    def __init__(self, model_name: str):
        """
        Initializes the model pipeline for explainability.

        Args:
            model_name (str): Hugging Face model name to load.
        """
        self.pipeline = pipeline("text-classification", model=model_name, tokenizer=model_name)
        print(f"Loaded model for explainability: {model_name}")

    def analyze_responses(self, prompts: List[str], responses: List[str]) -> List[Dict]:
        """
        Perform SHAP explainability on a list of prompts and responses.

        Args:
            prompts (List[str]): List of original prompts.
            responses (List[str]): List of corresponding model responses.

        Returns:
            List[Dict]: SHAP values and feature contributions for each response.
        """
        explainer = shap.Explainer(self.pipeline)
        shap_values = explainer(responses)
        explanations = []

        for i, response in enumerate(responses):
            explanations.append({
                "prompt": prompts[i],
                "response": response,
                "shap_values": shap_values[i].values.tolist(),
                "feature_contributions": shap_values[i].data.tolist()
            })

        return explanations

    def visualize_explanation(self, shap_values, feature_names):
        """
        Generates a SHAP summary plot for visualization.

        Args:
            shap_values: SHAP values for the features.
            feature_names: Names of the features being analyzed.
        """
        shap.summary_plot(shap_values, feature_names)

def save_explanations_to_file(explanations: List[Dict], output_path: str):
    """
    Save SHAP explanations to a JSON file.

    Args:
        explanations (List[Dict]): List of SHAP explanations.
        output_path (str): Path to save the explanations.
    """
    with open(output_path, "w") as file:
        json.dump(explanations, file, indent=4)
    print(f"Explanations saved to {output_path}")

if __name__ == "__main__":
    """
    Entry point for Explainable Analysis.

    What We Did:
    - Loaded model results from the referee LLM.
    - Used SHAP explainability to analyze feature contributions and decisions.
    - Visualized SHAP values to interpret model outputs.

    What's Next:
    - Refine weights and priors for models based on insights from the explanations.
    - Build interactive dashboards for real-time exploration of SHAP outputs.
    - Integrate explainability insights into RAG workflows to enhance transparency.
    """
    # Configurations
    config = ExplainableConfig(
        referee_results_path="./referee_results.json",
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        output_path="./explainable_analysis_results.json"
    )

    # Initialize Explainable Analysis
    explainable_analysis = ExplainableAnalysis(config.model_name)

    # Load Referee Results
    referee_results = load_referee_results(config.referee_results_path)

    # Analyze responses
    explanations = explainable_analysis.analyze_responses(
        referee_results["prompt"].tolist(),
        referee_results["final_response"].tolist()
    )

    # Save SHAP explanations
    save_explanations_to_file(explanations, config.output_path)

    # Visualize feature contributions (optional)
    print("Generating visualization...")
    explainable_analysis.visualize_explanation(
        shap_values=[e["shap_values"] for e in explanations],
        feature_names=referee_results["prompt"].tolist()
    )
