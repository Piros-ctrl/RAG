from src.models import AnsweredQuestion, MinimalSource
from search import Retriever
import json


class Answer:

    def __init__(self, qwen, chunks, query, question_id):
        self.chunks = chunks
        self.query = query
        self.qwen = qwen
        self.question_id = question_id
        self.eos = {self.qwen.encode("<|im_end|>")[0].tolist()[0],
                    self.qwen.encode("<|endoftext|>")[0].tolist()[0]
                    }

    def _prompt(self):
            context_list = [content["page_content"] for content in self.chunks]
            context = "\n\n----\n\n".join(context_list)
            return (
                    "You are a question-answering system.\n"
                    "Answer the question using ONLY the information contained in the context below.\n"
                    "Rules:\n"
                    "- Do not use your own knowledge.\n"
                    "- Do not use information from outside the context.\n"
                    "- Do not make assumptions or infer facts that are not explicitly supported.\n"
                    "- If the answer cannot be found in the context, say:\n"
                    "I don't know based on the provided context.\n\n"
                    "Context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{self.query}\n\n"
                    "Answer:"
                    )

    def _tokiniz(self):
        text = (
            "<|im_start|>user\n"
            f"{self._prompt()}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        return self.qwen.encode(text).tolist()[0]

    def generate_answer(self):
        encoded_prompt = self._tokiniz()
        respond = ""
        while 1:
            logits = self.qwen.get_logits_from_input_ids(encoded_prompt)
            next_token = logits.index(max(logits))
            encoded_prompt.append(next_token)
            respond += self.qwen.decode([next_token])
            if next_token in self.eos:
                 break
        return respond
