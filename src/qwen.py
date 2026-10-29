import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM_Model:
    """Wrap a Hugging Face causal language model and its tokenizer."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the tokenizer and model, and set the model to eval mode."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
                                                            model_name,
                                                            dtype="auto")
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """Convert text to a tensor of token ids with shape (1, length)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Convert token ids back to text, skipping special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Raw next-token logits (no softmax) for the given token ids."""
        x = torch.tensor([input_ids])
        with torch.inference_mode():
            logits = self.model(input_ids=x).logits[0, -1]
        return logits.float().tolist()
