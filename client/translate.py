import argparse
import asyncio
import json
import socket
import ssl
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

cwd_path: Path = Path(__file__).parent.resolve()

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


def translate_data(doc_content: bytes, url: str, cert_path: Optional[str] = None):
    headers = {"Authorization": "Bearer JWT_TOKEN"}
    data = {
        "doc": doc_content,
        "src_lang": "en",
        "tgt_lang": "fr",
    }
    try:
        response: requests.Response = requests.post(
            f"{url}/translate",
            data=data,
            headers=headers,
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
    response = translate_data(open(doc_path, "rb").read(), url)

    print("Response:", response["translation"])


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