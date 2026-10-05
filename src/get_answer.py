from llm_sdk.llm_sdk import Small_LLM_Model


class Answer:

    def __init__(self, chunks, query):
        self.chunks = chunks
        self.query = query
        self.qwen = Small_LLM_Model()

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
        return self.qwen.encode(self._prompt()).tolist()[0]

    def generate_answer(self):
        encoded_prompt = self._tokiniz()
        i = 0
        respond = ""
        while i < 40:
            logits = self.qwen.get_logits_from_input_ids(encoded_prompt)
            final_token = logits.index(max(logits))
            encoded_prompt.append(final_token)
            respond += self.qwen.decode([final_token])
            i += 1
        return respond

    def result(self):
        return self.generate_answer()