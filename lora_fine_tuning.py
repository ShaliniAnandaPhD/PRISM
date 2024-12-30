import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader

class LoRAFineTuningConfig:
    """
    Configuration for LoRA Fine-Tuning.
    """
    model_name: str  # Name of the pre-trained model
    dataset_name: str  # Hugging Face dataset name
    output_dir: str  # Directory to save the fine-tuned model
    lora_r: int = 8  # Low-rank dimension
    lora_alpha: int = 16  # Scaling factor for LoRA
    lora_dropout: float = 0.1  # Dropout for LoRA
    batch_size: int = 16  # Batch size for training
    num_epochs: int = 3  # Number of epochs for training
    learning_rate: float = 5e-5  # Learning rate

class LoRAFineTuning:
    """
    Implements Low-Rank Adaptation (LoRA) fine-tuning for domain-specific tasks.
    """

    def __init__(self, config: LoRAFineTuningConfig):
        """
        Initialize the fine-tuning process.

        Args:
            config (LoRAFineTuningConfig): Configuration for fine-tuning.
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        print(f"Model {config.model_name} loaded successfully.")

    def prepare_lora(self):
        """
        Prepare the model with LoRA configurations.
        """
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
        )
        self.model = get_peft_model(self.model, peft_config)
        print("LoRA configuration applied successfully.")

    def load_dataset(self):
        """
        Load and preprocess the dataset for fine-tuning.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        dataset = load_dataset(self.config.dataset_name, split="train")
        dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=128
            ),
            batched=True,
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        print(f"Dataset {self.config.dataset_name} loaded and preprocessed.")
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

    def train(self, dataloader):
        """
        Train the model using LoRA fine-tuning.

        Args:
            dataloader (DataLoader): DataLoader for the training dataset.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.model.device),
                    attention_mask=batch["attention_mask"].to(self.model.device),
                    labels=batch["input_ids"].to(self.model.device),
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {epoch_loss:.4f}")

    def save_model(self):
        """
        Save the fine-tuned model to the output directory.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}.")

if __name__ == "__main__":
    """
    Entry point for LoRA fine-tuning script.

    What We Did:
    - Loaded a pre-trained model and prepared it with LoRA configurations.
    - Loaded and preprocessed the dataset for domain-specific adaptation.
    - Fine-tuned the model using LoRA techniques for efficient adaptation.
    - Saved the fine-tuned model to the specified directory.

    What's Next:
    - Evaluate the fine-tuned model on domain-specific tasks.
    - Integrate the fine-tuned model into the Retrieval-Augmented Generation pipeline.
    - Experiment with other LoRA configurations for further optimization.
    """
    # Define configuration
    config = LoRAFineTuningConfig(
        model_name="gpt2",
        dataset_name="lex_glossary",  # Updated dataset name to reflect legal context
        output_dir="./lora_fine_tuned_model",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        batch_size=16,
        num_epochs=3,
        learning_rate=5e-5,
    )

    # Initialize and run fine-tuning
    lora_fine_tuner = LoRAFineTuning(config)
    lora_fine_tuner.prepare_lora()
    dataloader = lora_fine_tuner.load_dataset()
    lora_fine_tuner.train(dataloader)
    lora_fine_tuner.save_model()
