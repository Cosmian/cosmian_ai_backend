# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

## API routes

* `/kms_summarize`

Input

```js
multipart
form: {
    key_id: "str",
    nonce: "base64",
}

file: "encrypted_doc"
```

Output

```js
{
    encrypted_summary: "base64",
    nonce: "base64",
}
```

* `/client_summarize`

Request:

```js

form: {
    doc: "str",
}
```

Response:

```js
{
    summarize: "str"
}
```
