from model_pipeline import ModelPipeline
from transformers import AutoModelForSeq2SeqLM


class Summarizer(ModelPipeline):
    model_class = AutoModelForSeq2SeqLM

    def __init__(self, model_name: str, min_tokens_length=30, max_tokens_length=100):
        super().__init__()

        self.model_name = model_name
        self.min_tokens_length = min_tokens_length
        self.max_tokens_length = max_tokens_length

    def encode(self, text):
        """Additional checks on the size of the input"""
        preprocess_text = text.strip().replace("\n", "")
        input_tokens = self.tokenizer.encode(
            preprocess_text, return_tensors="pt", truncation=True
        )
        if len(input_tokens[0]) < self.min_tokens_length:
            raise ValueError("Input text too short to summarize")

        return input_tokens

    def forward(self, input_tokens):
        return self.model.generate(
            input_tokens,
            min_length=self.min_tokens_length,
            max_length=self.max_tokens_length,
            do_sample=False,
        )[0]

    def decode(self, output_tokens):
        return self.tokenizer.decode(output_tokens, skip_special_tokens=True)
