import os
from http import HTTPStatus

import torch
from auth import check_token
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from summarizer import Summarizer
from translator import Translator

torch.set_num_threads(os.cpu_count() or 1)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "facebook/bart-large-cnn")
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.5"))
summarizer = Summarizer(model=SUMMARY_MODEL, temperature=SUMMARY_TEMPERATURE)


TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
translator = Translator(model=TRANSLATION_MODEL)


@app.post("/summarize")
@check_token()
async def post_summarize():
    if "doc" not in request.form:
        return ("Error: Missing file content", 400)
    text = request.form["doc"]

    try:
        summary = summarizer(text)
    except ValueError as e:
        return (str(e), 400)

    return jsonify(
        {
            "summary": summary,
        }
    )


@app.post("/translate")
@check_token()
async def post_translate():
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
    except ValueError as e:
        return (f"Error: {e}", 400)

    return jsonify(
        {
            "translation": result,
        }
    )


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)
