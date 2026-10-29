from typing import Any


class Answer:
    """Build a prompt from retrieved chunks & generate an answer with Qwen."""

    def __init__(
        self,
        qwen: Any,
        chunks: list[dict[str, Any]],
        query: str,
        question_id: str
    ) -> None:
        """Store the model, retrieved chunks, question and its id."""
        self.chunks = chunks
        self.query = query
        self.qwen = qwen
        self.question_id = question_id
        self.eos = {self.qwen.encode("<|im_end|>")[0].tolist()[0],
                    self.qwen.encode("<|endoftext|>")[0].tolist()[0]
                    }

    def _prompt(self) -> str:
        """Return the question-answering prompt containing the context."""
        context_list = [content["page_content"] for content in self.chunks]
        context = "\n\n----\n\n".join(context_list)
        return (
            "You are a question-answering system.\n"
            "Answer the question using ONLY the information "
            "contained in the context below.\n"
            "Rules:\n"
            "- Do not use your own knowledge.\n"
            "- Do not use information from outside the context.\n"
            "- Do not make assumptions or infer facts "
            "that are not explicitly supported.\n"
            "- If the answer cannot be found in the context, say:\n"
            "I don't know based on the provided context.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{self.query}\n\n"
            "Answer:"
                )

    def _tokiniz(self) -> list[int]:
        """Wrap the prompt in the chat template and convert it to token ids."""
        text = (
            "<|im_start|>user\n"
            f"{self._prompt()}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        return self.qwen.encode(text).tolist()[0]

    def generate_answer(self) -> str:
        """Generate the answer token by token until an end token appears."""
        encoded_prompt = self._tokiniz()
        respond = ""
        while 1:
            logits = self.qwen.get_logits(encoded_prompt)
            next_token = logits.index(max(logits))
            encoded_prompt.append(next_token)
            respond += self.qwen.decode([next_token])
            if next_token in self.eos:
                break
        return respond
