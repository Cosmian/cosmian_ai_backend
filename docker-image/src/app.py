import json
import os
from base64 import b64decode, b64encode
from http import HTTPStatus
from pathlib import Path

from cosmian_kms import KmsClient
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from transformers import AutoTokenizer, pipeline

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_NAME = os.getenv("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
summarizer = pipeline("summarization", model=MODEL_NAME)


KMS_URL = os.getenv("KMS_URL")
KMS_API_KEY = os.getenv("KMS_API_KEY")
client = KmsClient(
    KMS_URL,
    api_key=KMS_API_KEY,
)


def summarize(text: str, min_summary_length=30, max_summary_length=100) -> str:
    # Preprocess
    preprocess_text = text.strip().replace("\n", "")
    input_tokens_length = len(tokenizer.encode(preprocess_text, return_tensors="pt")[0])

    if input_tokens_length < min_summary_length:
        raise ValueError("Input text too short to summarize")

    # Summarize
    output = summarizer(
        preprocess_text,
        min_length=min_summary_length,
        max_length=max_summary_length,
        do_sample=True,
        clean_up_tokenization_spaces=True,
    )

    return output[0]["summary_text"]


@app.post("/kms_summarize")
async def kms_summarize():
    if not "key_id" in request.form:
        return ("Error: Missing key id", 400)
    if not "nonce" in request.form:
        return ("Error: Missing nonce", 400)

    # Retrieve Key and Nonce
    try:
        key = await client.get_object(request.form["key_id"])
    except Exception as e:
        return (f"Error getting key from KMS: {e}", 400)

    nonce = b64decode(request.form["nonce"])
    aes = AES.new(key.key_block(), AES.MODE_GCM, nonce)

    if "encrypted_doc" not in request.files:
        return ("Error: Missing file content", 400)

    # Read file from client
    ciphertext = request.files["encrypted_doc"].read()

    # Decrypt doc
    text = aes.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:]).decode("utf-8")

    try:
        summary = summarize(text)
    except ValueError:
        return ("Error: Input text too short to summarize", 400)

    # Encrypt output
    nonce = get_random_bytes(12)
    aes = AES.new(key.key_block(), AES.MODE_GCM, nonce)
    enc_output, tag = aes.encrypt_and_digest(summary.encode("utf-8"))

    return jsonify(
        {
            "nonce": b64encode(nonce).decode("utf-8"),
            "encrypted_summary": b64encode(enc_output + tag).decode("utf-8"),
        }
    )


@app.post("/client_summarize")
async def client_summarize():
    # Get text from client
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


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)
