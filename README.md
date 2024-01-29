# demo-cse-cc

Confidential computing demo on how to use LLMs on GWS encrypted docs

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
