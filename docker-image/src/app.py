import os
from http import HTTPStatus

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from transformers import AutoTokenizer, pipeline
from transformers.tools import TranslationTool

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL")
summary_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
summarizer = pipeline("summarization", model=SUMMARY_MODEL)

TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL")
translator = TranslationTool(model=TRANSLATION_MODEL)


def summarize(text: str, min_summary_length=30, max_summary_length=100) -> str:
    # Preprocess
    preprocess_text = text.strip().replace("\n", "")
    input_tokens_length = len(
        summary_tokenizer.encode(preprocess_text, return_tensors="pt")[0]
    )

    if input_tokens_length < min_summary_length:
        raise ValueError("Input text too short to summarize")

    # Summarize
    output = summarizer(
        preprocess_text,
        min_length=min_summary_length,
        max_length=max_summary_length,
        do_sample=True,
        clean_up_tokenization_spaces=True,
        truncation=True,
    )

    return output[0]["summary_text"]


@app.post("/summarize")
async def post_summarize():
    # TODO: check JWT

    if "doc" not in request.form:
        return ("Error: Missing file content", 400)
    text = request.form["doc"]

    try:
        summary = summarize(text)
    except ValueError:
        return ("Error: Input text too short to summarize", 400)

    return jsonify(
        {
            "summary": summary,
        }
    )


@app.post("/translate")
async def post_translate():
    # TODO: check JWT

    if "doc" not in request.form:
        return ("Error: Missing file content", 400)
    if "src_lang" not in request.form:
        return ("Error: Missing source language", 400)
    if "tgt_lang" not in request.form:
        return ("Error: Missing target language", 400)

    text = request.form["doc"]
    src_lang = request.form["src_lang"]
    tgt_lang = request.form["tgt_lang"]

    try:
        result = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
    except ValueError:
        return ("Error: Input text too short to summarize", 400)

    return jsonify(
        {
            "result": result,
        }
    )


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)
