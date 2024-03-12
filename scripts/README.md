# Benchmark and interact with the API

## Sample data

- `sample_en_doc.txt`: sample english documentation from Cloudproof (~7000 characters)

- `sample_fr_doc.txt`: extract of a French tale (~4000 characters)

## Clients

Use the API to perform summarize or translation.

See [./client](./client) for instructions.

## Benchmark

### `bench_inference.py`

Directly measure inference time on the machine it is executed.

```bash
python bench_inference.py sample_data/sample_en_doc.txt [-n 2] [--verbose]
```

### `bench_api.py`

Measure and plots the responses time of the requests to the API.

```bash
python bench_api.py URL [--translate] [--save-plot "./bench_plots/plot_name.jpg"]
```

- `bench_plots`

### Query time plots using `bench_api.py`

- `inference` different hardware: CPU, CPU with AMX, CUDA

- `web_server`: try parallelizing the request processing, at the end we are still limited by the text generation using all computing capabilities (maybe useful in the future with multiple GPUs)
