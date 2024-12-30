import os
import re
from typing import List
import pandas as pd  # For tabular data management
from pydantic import BaseModel, DirectoryPath, FilePath
from transformers import AutoTokenizer

# Load a pre-trained tokenizer (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class Config(BaseModel):
    """
    Configuration model to validate input and output directories.
    """
    input_dir: DirectoryPath
    output_dir: DirectoryPath

    def ensure_output_dir(self):
        """
        Ensures the output directory exists. If not, it creates one.
        """
        os.makedirs(self.output_dir, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Cleans the input text by removing URLs, special characters, and normalizing whitespace.

    Args:
        text (str): The raw text input.

    Returns:
        str: The cleaned text.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes the input text into subword tokens using a pre-trained tokenizer.

    Args:
        text (str): The cleaned text input.

    Returns:
        List[str]: A list of tokens.
    """
    # Tokenize using a pre-trained tokenizer
    tokens = tokenizer.tokenize(text)
    return tokens


def process_file(file_path: FilePath, output_dir: str):
    """
    Processes a single text file: cleans and tokenizes its content, then saves the results.

    Args:
        file_path (FilePath): Path to the raw text file.
        output_dir (str): Directory to save the processed file.
    """
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name.replace(".txt", "_processed.csv"))

    # Read raw text
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Clean the text
    cleaned_text = clean_text(raw_text)

    # Tokenize the text
    tokens = tokenize_text(cleaned_text)

    # Save the tokens in a tabular format
    df = pd.DataFrame(tokens, columns=["Tokens"])
    df.to_csv(output_file_path, index=False)

    print(f"Processed and saved: {file_name}")


def process_directory(config: Config):
    """
    Processes all text files in a directory by cleaning and tokenizing their contents.

    Args:
        config (Config): Configuration object containing directory paths.
    """
    # Ensure the output directory exists
    config.ensure_output_dir()

    # Iterate over all files in the input directory
    for file_name in os.listdir(config.input_dir):
        file_path = os.path.join(config.input_dir, file_name)

        # Skip non-text files
        if not file_name.endswith(".txt"):
            print(f"Skipping non-text file: {file_name}")
            continue

        # Process the text file
        process_file(file_path, config.output_dir)


if __name__ == "__main__":
    """
    Entry point for the script. Defines paths and starts processing.
    """
    # Define paths (adjust these paths as needed)
    input_directory = "./raw_data"
    output_directory = "./processed_data"

    # Validate configuration
    config = Config(input_dir=input_directory, output_dir=output_directory)

    # Process the directory
    process_directory(config)

# ---------------------------------------------------------------
# What We Did:
# ---------------------------------------------------------------
# - Used Pydantic to validate and manage directory paths, ensuring clean and robust input handling.
# - Integrated Hugging Face's BERT tokenizer for state-of-the-art subword tokenization.
# - Saved cleaned and tokenized text as CSV files, making them easy to use in other workflows.
# - Broke the functionality into modular functions for better readability and reuse.
# ---------------------------------------------------------------
# What's Next:
# ---------------------------------------------------------------
# - Add parallel processing with tools like Dask to handle larger datasets more efficiently.
# - Expand features with NLP capabilities like entity extraction or sentiment analysis.
# - Build a FastAPI endpoint to turn this into a web-accessible service.
# ---------------------------------------------------------------
# 1. Pydantic is used to validate and manage directory configurations, ensuring robust input handling.
# 2. The Hugging Face tokenizer (BERT) is leveraged for subword-level tokenization, providing state-of-the-art NLP capabilities.
# 3. Cleaned and tokenized text is saved as CSV files for easy integration with downstream workflows.
# 4. Modular functions are used for cleaning, tokenization, file processing, and directory processing to enhance readability and reusability.
# ---------------------------------------------------------------
# Next Steps:
# ---------------------------------------------------------------
# - Add parallel processing using tools like Dask to handle large datasets efficiently.
# - Implement additional NLP features, such as entity extraction or sentiment analysis, for enriched outputs.
# - Create a FastAPI endpoint to expose this functionality as a web service.
# ---------------------------------------------------------------
