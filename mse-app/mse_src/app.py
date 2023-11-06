import os
from http import HTTPStatus

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
cors = CORS(app)

CWD_PATH = os.getenv("MODULE_PATH")
model = AutoModelForSeq2SeqLM.from_pretrained(f"{CWD_PATH}/t5-base-model")
tokenizer = AutoTokenizer.from_pretrained(f"{CWD_PATH}/t5-base-tokenizer")


@app.post("/summarize")
def summarize():
    if not "key" in request.form:
        return ("Error: Missing key", 400)

    key = request.form["key"]
    aes = AES.new(key.encode("utf-8"), AES.MODE_CBC, b"a" * 16)

    if "encrypted_doc" not in request.files:
        return ("Error: Missing file", 400)

    # Read file from client
    ciphertext = request.files["encrypted_doc"].read()
    print("Ciphertext len:", len(ciphertext))

    # Decrypt here
    text = unpad(aes.decrypt(ciphertext), aes.block_size).decode("utf-8")
    print("Cleartext len:", len(text))

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
    return jsonify({"generated_text": output})


@app.get("/health")
def health_check():
    """Health check of the application."""
    return Response(response="OK", status=HTTPStatus.OK)
