# Cosmian AI backend

Confidential computing backend to run language models

## Usage

- Build and install the [app](./app/README.md)

- Edit the config file ([more info](./app/README.md#config-file))

- Run the app

```bash
CONFIG_PATH="./run/config.json" cosmian-ai-runner --port 5001
```

- Create a `supervisord` service:

Sample config file: [./run/cosmian-ai-backend.conf](./run/cosmian-ai-backend.conf)

## API endpoints

`/summarize`

Request:

- `doc`: content of the document to summarize (String)

Response:

- `summary`: result (String)

`/translate`

Request:

- `doc`: content of the document to translate (String)

- `src_lang`: source language of the text to summarize (String),

- `tgt_lang`: desired output language (String),

Response:

- `translation`: result (String)
