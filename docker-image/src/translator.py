import torch
from transformers.tools import TranslationTool


def find_separator_index(input_tokens, sep_token, start_index):
    split_index = start_index
    while input_tokens[split_index] != sep_token:
        split_index -= 1
        if split_index == 0:
            return start_index
    return split_index


def end_pad_tokens(input_tokens, length, pad_token, eos_token):
    input_tokens.extend([pad_token] * (length - len(input_tokens) - 1))
    input_tokens.append(eos_token)
    assert len(input_tokens) == length
    return input_tokens


class Translator(TranslationTool):
    def __init__(self, chunk_size=500, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens_length = chunk_size

    def split_chunks(self, input_tokens, bos_token, eos_token):
        if len(input_tokens) <= self.max_tokens_length:
            return torch.tensor([input_tokens])

        sep_token = self.pre_processor.encode(".", add_special_tokens=False)[0]

        chunks = []
        # Split at the previous `sep` closer to `max_tokens_length`
        while len(input_tokens) > self.max_tokens_length:
            split_index = find_separator_index(
                input_tokens, sep_token, self.max_tokens_length - 1
            )
            chunks.append(
                end_pad_tokens(
                    input_tokens[:split_index],
                    self.max_tokens_length,
                    self.pre_processor.pad_token_id,
                    eos_token,
                )
            )
            input_tokens = input_tokens[split_index:]
            input_tokens[0] = bos_token

        # Add last chunk
        chunks.append(
            end_pad_tokens(
                input_tokens[:split_index],
                self.max_tokens_length,
                self.pre_processor.pad_token_id,
                eos_token,
            )
        )
        return torch.tensor(chunks)

    def forward(self, inputs):
        input_tokens = inputs["input_ids"][0].tolist()
        bos_token = input_tokens[0]
        eos_token = input_tokens[-1]

        input_ids = self.split_chunks(input_tokens, bos_token, eos_token)

        return self.model.generate(
            input_ids, forced_bos_token_id=inputs["forced_bos_token_id"]
        )

    def decode(self, outputs):
        decoded_batch = self.post_processor.batch_decode(
            outputs, skip_special_tokens=True
        )
        return " ".join(decoded_batch)
