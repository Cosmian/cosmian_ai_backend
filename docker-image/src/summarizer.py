from transformers.tools import TextSummarizationTool


class Summarizer(TextSummarizationTool):
    def __init__(self, min_tokens_length=30, max_tokens_length=100, **kwargs):
        super().__init__(**kwargs)
        self.min_tokens_length = min_tokens_length
        self.max_tokens_length = max_tokens_length

    def encode(self, text):
        """Additional checks on the size of the input"""
        preprocess_text = text.strip().replace("\n", "")
        input_tokens = self.pre_processor.encode(
            preprocess_text, return_tensors="pt", truncation=True
        )
        if len(input_tokens[0]) < self.min_tokens_length:
            raise ValueError("Input text too short to summarize")

        return input_tokens

    def forward(self, inputs):
        return self.model.generate(
            **inputs,
            min_length=self.min_tokens_length,
            max_length=self.max_tokens_length,
            do_sample=True
        )[0]
