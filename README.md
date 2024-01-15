# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

## API routes

* `/summarize`

Request:

```js
form: {
    doc: "str"
}
```

Response:

```js
{
    summarize: "str"
}
```

* `/translate`

Request:

```js
form: {
    src_lang: "str",
    tgt_lang: "str",
    doc: "str"
}
```

Response:

```js
{
    result: "str"
}
```