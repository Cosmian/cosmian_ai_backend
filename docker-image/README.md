# API to run a NLP model in a secure enclave

## Build and use local image

```bash
docker build -t local/demo-cse-cc .
```

* Run

```bash
docker run -it -p 5000:5000 local/demo-cse-cc
```

## MSE Usage

* Run flask server

```bash
mse cloud localtest --no-tests
```

* Run client

```bash
python3 client/client.py http://localhost:5000
```
