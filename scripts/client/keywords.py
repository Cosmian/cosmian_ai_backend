# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
from pathlib import Path
from typing import List

import requests

cwd_path: Path = Path(__file__).parent.resolve()


def extract_keywords(doc_content: bytes, url: str):
    data = {"doc": doc_content, "src_lang": "en"}
    try:
        response: requests.Response = requests.post(
            f"{url}/extract",
            data=data,
        )
    except requests.exceptions.SSLError:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    if response.status_code != 200:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    return json.loads(response.text)


def index_keywords(keywords: List[str], url: str):
    data = {"keywords": json.dumps(keywords), "src_lang": "en"}
    try:
        response: requests.Response = requests.post(
            f"{url}/index",
            data=data,
        )
    except requests.exceptions.SSLError:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    if response.status_code != 200:
        raise Exception(
            f"Bad response from server: {response.status_code} {response.text}"
        )

    return json.loads(response.text)


async def main(url: str, doc_path: str, self_signed_ssl: bool = False):
    response = extract_keywords(open(doc_path, "rb").read(), url)
    print("/extract response:", response)

    response = index_keywords(response["keywords"], url)
    print("/index response:", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data provider example.")
    parser.add_argument("url", type=str, help="URL of the secure API")
    parser.add_argument("doc", type=str, help="The document to process")

    try:
        args = parser.parse_args()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(args.url, args.doc))
    except SystemExit:
        parser.print_help()
        raise
