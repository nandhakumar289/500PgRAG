from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModel
import torch
from dataclasses import dataclass, field


@dataclass
class CustomEmbedding(HuggingFaceEmbedding):

    # Load the tokenizer and model`
    tokenizer: AutoTokenizer = field(default=None)
    model: AutoModel = field(default=None)

    def __init__(
        self, tokenizer: AutoTokenizer, model: AutoModel, model_name: str
    ) -> None:
        """
        Initialize the CustomEmbedding class.
        Args:
            tokenizer (AutoTokenizer): Tokenizer instance.
            model (AutoModel): Model instance.
            model_name (str): Model name.
        """
        # Initialize the parent class with the model_name
        super().__init__(model_name=model_name)
        self.tokenizer = tokenizer
        self.model = model

    def _get_text_embedding(self, text: str) -> list:
        """
        Generate embedding for a single text string.
        Args:
            text (str): Input text.
        Returns:
            list: Embedding vector.
        """
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the CLS token's embedding as the text embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).tolist()
        return embedding
