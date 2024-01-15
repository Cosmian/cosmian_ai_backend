# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

## API endpoints

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
    summary: "str"
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
    translation: "str"
}
```
