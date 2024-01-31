import torch
from transformers.tools import TranslationTool


def find_separator_index(input_tokens, sep_token, start_index):
    split_index = start_index
    while input_tokens[split_index] != sep_token:
        split_index -= 1
        if split_index == 0:
            return start_index
    return split_index


def end_pad_tokens(input_tokens, length, pad_token, eos_token=None):
    if eos_token:
        input_tokens.append(eos_token)
    input_tokens.extend([pad_token] * (length - len(input_tokens)))
    assert len(input_tokens) == length
    return input_tokens


# ISO 639 language codes based on `https://en.wikipedia.org/wiki/Languages_used_on_the_Internet`
LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "nb": "nno_Latn",
    "nl": "nld_Latn",
    "nn": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


class Translator(TranslationTool):
    def __init__(self, chunk_size=150, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens_length = chunk_size
        self.lang_to_code = LANGUAGE_CODES

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
                input_tokens,
                self.max_tokens_length,
                self.pre_processor.pad_token_id,
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
        return "\n".join(decoded_batch)
