import json
from typing import List, Dict
from pydantic import BaseModel
import openai

class RefereeConfig(BaseModel):
    """
    Configuration for the Referee LLM approach.
    """
    evaluation_results_path: str  # Path to the evaluation results JSON file
    referee_api_key: str  # API key for the referee LLM
    referee_model: str  # Referee model to use (e.g., GPT-4)
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

def query_referee_llm(prompt: str, responses: List[str], api_key: str, model: str) -> str:
    """
    Queries the Referee LLM to resolve conflicts among responses.

    Args:
        prompt (str): The original input prompt.
        responses (List[str]): Responses from different models to be evaluated.
        api_key (str): API key for accessing the referee LLM.
        model (str): Referee model to use (e.g., GPT-4).

    Returns:
        str: The final response chosen by the Referee LLM.
    """
    openai.api_key = api_key
    combined_prompt = f"""
    Original Prompt: {prompt}

    Candidate Responses:
    1. {responses[0]}
    2. {responses[1]}
    3. {responses[2]}

    Please evaluate the above responses and choose the most contextually accurate and coherent one. Provide your reasoning as well.
    """
    response = openai.Completion.create(
        model=model,
        prompt=combined_prompt,
        max_tokens=300,
        temperature=0.7
    )
    chosen_response = response.choices[0].text.strip()
    return chosen_response

def process_referee_aggregation(config: RefereeConfig):
    """
    Processes evaluation results and uses a Referee LLM to resolve conflicts among model outputs.

    Args:
        config (RefereeConfig): Configuration for the referee approach.
    """
    # Load evaluation results
    results = load_evaluation_results(config.evaluation_results_path)

    # Initialize aggregated results
    aggregated_results = []

    # Process each prompt and its responses
    for i, prompt in enumerate(results["prompts"]):
        responses = [results["openai"][i], results["mistral"][i], results["claude"][i]]

        # Query the Referee LLM
        final_response = query_referee_llm(prompt, responses, config.referee_api_key, config.referee_model)

        # Collect results
        aggregated_results.append({
            "prompt": prompt,
            "final_response": final_response,
            "responses": {
                "openai": responses[0],
                "mistral": responses[1],
                "claude": responses[2],
            },
        })

    # Save aggregated results
    with open(config.output_path, "w") as file:
        json.dump(aggregated_results, file, indent=4)
    print(f"Aggregated results saved to {config.output_path}")

if __name__ == "__main__":
    """
    Entry point for the Referee LLM script.

    What We Did:
    - Loaded evaluation results:
      - Combined multiple model responses for each prompt.

    - Referee LLM Integration:
      - Queried a specialized Referee LLM (e.g., GPT-4) to resolve ambiguities.
      - Evaluated and selected the most accurate and coherent response.

    - Saved Aggregated Outputs:
      - Stored the final responses along with the original model outputs in `referee_results.json`.

    What's Next:
    - Explainable Analysis:
      - Use Explainable AI to understand the reasoning behind the Referee LLM's decisions.

    - Fine-Tuning with LoRA:
      - Adapt the Referee LLM for domain-specific tasks like legal or technical document review.

    - Integrate into RAG Pipeline:
      - Embed referee decisions into the Retrieval-Augmented Generation workflow for enhanced output quality.
    """
    # Define the configuration
    config = RefereeConfig(
        evaluation_results_path="./model_evaluation_results.json",
        referee_api_key="your-openai-api-key",
        referee_model="text-davinci-003",
        output_path="./referee_results.json",
    )

    # Run the Referee aggregation process
    process_referee_aggregation(config)
