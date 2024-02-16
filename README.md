# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

## Usage

* Build python [app](./app/README.md)

* Run app in `./staging` or `./prod`

```bash
./run.sh
```

* Create a `supervisord` service:

Sample config file: [./staging/cosmian_confidential_ai.conf](./staging/cosmian_confidential_ai.conf)

## API endpoints

`/summarize`

Request:

* `doc`: content of the document to summarize (String)

Response:

* `summary`: result (String)

`/translate`

Request:

* `doc`: content of the document to translate (String)

* `src_lang`: source language of the text to summarize (String),

* `tgt_lang`: desired output language (String),

Response:

* `translation`: result (String)
