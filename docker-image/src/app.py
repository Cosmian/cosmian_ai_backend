import json
import os
from http import HTTPStatus
from pathlib import Path

import torch
from auth import check_token
from config import AppConfig
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from summarizer import Summarizer
from translator import Translator

torch.set_num_threads(os.cpu_count() or 1)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

with open("config.json") as f:
    app_config = AppConfig.load(f)

summarizer = Summarizer(
    model=app_config["summary"]["model"],
    temperature=float(app_config["summary"]["temperature"]),
)
translator = Translator(model=app_config["translation"]["model"])


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
