import argparse
import asyncio
import json
import socket
import ssl
import tempfile
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from cosmian_kms import KmsClient
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

cwd_path: Path = Path(__file__).parent.resolve()
ENCRYPTED_DOC_PATH = cwd_path / "doc.enc"
KEY_ID = "e06c1896-7b99-47c6-a7f2-69e2052f6e4a"

# read KMS API key from secret file
SECRETS = json.loads((cwd_path.parent / "secrets.json").read_text(encoding="utf-8"))
client = KmsClient(
    "https://developer-example.cosmian.com/kms", api_key=SECRETS["kms_api_key"]
)


def get_certificate(hostname: str, port: int) -> str:
    with socket.create_connection((hostname, port)) as sock:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            bin_cert = ssock.getpeercert(True)
            if not bin_cert:
                raise Exception("Can't get peer certificate")
            return ssl.DER_cert_to_PEM_cert(bin_cert)


def summarize_data(
    encrypted_doc_path: Path, nonce: bytes, url: str, cert_path: Optional[Path] = None
):
    files = {"encrypted_doc": open(encrypted_doc_path, "rb")}
    data = {"key_id": KEY_ID, "nonce": b64encode(nonce)}
    try:
        response: requests.Response = requests.post(
            f"{url}/kms_summarize",
            files=files,
            data=data,
            verify=cert_path,
        )
    except requests.exceptions.SSLError as e:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    if response.status_code != 200:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    return json.loads(response.text)


async def main(url: str, doc_path: str, self_signed_ssl: bool = False):
    parsed_url = urlparse(url)

    cert_path: Optional[Path] = None
    if self_signed_ssl and parsed_url.scheme == "https":
        hostname = parsed_url.hostname
        port = 443 if parsed_url.port is None else parsed_url.port

        cert_path = Path(tempfile.gettempdir()) / f"{hostname}.pem"
        cert_data = get_certificate(hostname, port)
        cert_path.write_bytes(cert_data.encode("utf-8"))

    # Encrypt doc
    nonce = get_random_bytes(12)
    key = await client.get_object(KEY_ID)
    aes = AES.new(key.key_block(), AES.MODE_GCM, nonce)
    with open(doc_path, "rb") as f:
        ciphertext, tag = aes.encrypt_and_digest(f.read())
    with open(ENCRYPTED_DOC_PATH, "wb") as f:
        f.write(ciphertext + tag)

    response = summarize_data(ENCRYPTED_DOC_PATH, nonce, url, cert_path)

    ciphertext = b64decode(response["encrypted_summary"])
    nonce = b64decode(response["nonce"])
    aes = AES.new(key.key_block(), AES.MODE_GCM, nonce)
    text = aes.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:]).decode("utf-8")
    print("Summary:", text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data provider example.")
    parser.add_argument("url", type=str, help="URL of the secure API")
    parser.add_argument("doc", type=str, help="The document to summarize")

    try:
        args = parser.parse_args()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(args.url, args.doc))
    except SystemExit:
        parser.print_help()
        raise
