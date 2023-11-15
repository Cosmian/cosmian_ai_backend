# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

## API routes

* `/kms_summarize`

Input

```json
multipart
data: {
    "key_id": "str",
    "nonce": "base64",
}

file: "encrypted_doc"
```

Output

```json
{
    "encrypted_summary": "base64",
    "nonce": "base64",
}
```

* `client_summarize`

Request:

```json
file: "doc"
```

Response:

```json
{
    "summarize": "str"
}
```
