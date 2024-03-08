# Benchmark and interact with the API

## Sample data

- `sample_en_doc.txt`: sample english documentation from Cloudproof (~7000 characters)

- `sample_fr_doc.txt`: extract of a French tale (~4000 characters)

## Clients

Use the API to perform summarize or translation.

See [./client](./client) for instructions.

## Benchmark

- `bench_inference.py`: measure inference time where it is executed

- [`client/bench_api.py`](./client/README.md)

- `bench_plots`: query time plot using `bench_api.py`
