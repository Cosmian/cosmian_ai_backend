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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

CWD_PATH = os.getenv("MODULE_PATH")
model = AutoModelForSeq2SeqLM.from_pretrained(f"{CWD_PATH}/t5-base-model")
tokenizer = AutoTokenizer.from_pretrained(f"{CWD_PATH}/t5-base-tokenizer")


# read KMS API key from secret file
SECRETS = json.loads(
    Path(os.getenv("SECRETS_PATH", "../secrets.json")).read_text(encoding="utf-8")
)
client = KmsClient(
    "https://developer-example.cosmian.com/kms",
    api_key=SECRETS["kms_api_key"],
    insecure_mode=True,
)


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
        return ("Error: Missing file", 400)

    # Read file from client
    ciphertext = request.files["encrypted_doc"].read()

    # Decrypt doc
    text = aes.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:]).decode("utf-8")

    # Preprocess and tokenize
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(
        t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True
    )
    # Summarize
    summary_tokens = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=200,
        early_stopping=True,
    )
    output = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)

    # Encrypt output
    nonce = get_random_bytes(12)
    aes = AES.new(key.key_block(), AES.MODE_GCM, nonce)
    enc_output, tag = aes.encrypt_and_digest(output.encode("utf-8"))

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
        return ("Error: Missing file", 400)
    text = request.form["doc"]

    # Preprocess and tokenize
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(
        t5_prepared_Text, return_tensors="pt", max_length=512, truncation=True
    )
    # Summarize
    summary_tokens = model.generate(
        tokenized_text,
        num_beams=4,
        no_repeat_ngram_size=2,
        min_length=30,
        max_length=200,
        early_stopping=True,
    )

    output = tokenizer.decode(summary_tokens[0], skip_special_tokens=True)
    return jsonify(
        {
            "summary": output,
        }
    )


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)
