# -*- coding: utf-8 -*-
import argparse
import time

import torch
from accelerate.utils import send_to_device
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.set_num_threads(8)
device = "cuda" if torch.cuda.is_available() else "cpu"


def bench_model(input_batch, translate=False, verbose=False):
    # Load model
    model_name = (
        "facebook/nllb-200-distilled-600M" if translate else "facebook/bart-large-cnn"
    )
    print(f"\n------ Benchmark {model_name} ------\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize
    if translate:
        tokenized_text = tokenizer._build_translation_inputs(
            input_batch,
            src_lang="eng_Latn",
            tgt_lang="fra_Latn",
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
    else:
        tokenized_text = tokenizer(
            input_batch, truncation=True, padding=True, return_tensors="pt"
        )

    # Move to GPU if available
    model.to(device)
    encoded_inputs = send_to_device(tokenized_text, device)

    # Benchmark
    start = time.time()
    res = model.generate(**encoded_inputs, min_length=250, max_length=250)
    duration = time.time() - start

    nb_generated_tokens = res.shape[0] * res.shape[1]
    print(
        f"Inference took {duration:.2f} seconds ({1000*duration/nb_generated_tokens:.2f} ms / tokens)"
    )

    if verbose:
        print("\n------ Generated response ------\n")
        for response_tokens in res:
            print(tokenizer.decode(response_tokens, skip_special_tokens=True))


def bench(input_file: str, nb_batch=1, verbose=False):
    # Load data
    with open(input_file) as f:
        text = f.read()

    # Create batch
    text_split = len(text) // nb_batch
    batch = []
    for i in range(nb_batch):
        batch.append(text[text_split * i : text_split * (i + 1)])

    print(f"Batch: {nb_batch} sample(s) of {text_split} characters")

    bench_model(batch, translate=False, verbose=verbose)
    bench_model(batch, translate=True, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="File containing sample text.")
    parser.add_argument(
        "-n",
        "--nb-batch",
        type=int,
        default=1,
        help="Split the sample text in N batches",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Output the generated results",
    )
    args = parser.parse_args()

    bench(args.input_file, args.nb_batch, args.verbose)
