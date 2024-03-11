# -*- coding: utf-8 -*-
import argparse
import threading
import time

import matplotlib.pyplot as plt
from client.summarize import summarize_data
from client.translate import translate_data


def timed(func):
    """
    records approximate durations of function calls
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        print("{name:<30} started".format(name=func.__name__))
        _ = func(*args, **kwargs)
        duration = time.time() - start
        duration_str = "{name:<30} finished in {elapsed:.2f} seconds".format(
            name=func.__name__, elapsed=duration
        )
        print(duration_str)
        return duration

    return wrapper


@timed
def bench_translate(url, nb_queries=10):
    content = open("sample_data/sample_en_doc.txt", "rb").read()

    print(f"Sending {nb_queries} queries of {len(content)} bytes")

    requests = []
    for _ in range(nb_queries):
        t = threading.Thread(target=translate_data, args=(content, url))
        t.start()
        requests.append(t)

    for t in requests:
        t.join()


@timed
def bench_summarize(url, nb_queries=10):
    content = open("sample_data/sample_en_doc.txt", "rb").read()

    print(f"Sending {nb_queries} queries of {len(content)} bytes")

    requests = []
    for _ in range(nb_queries):
        t = threading.Thread(target=summarize_data, args=(content, url))
        t.start()
        requests.append(t)

    for t in requests:
        t.join()


def main(url: str, translate=False, save_plot=None):

    f = bench_translate if translate else bench_summarize

    nb_requests = list(range(1, 10, 2))
    durations = []
    for i in nb_requests:
        durations.append(f(url, i))

    if save_plot:
        plt.scatter(nb_requests, durations)
        plt.xlabel("Number of requests")
        plt.ylabel("Response time")
        plt.title("Translate" if translate else "Summarize")
        plt.savefig(save_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="URL of the API to benchmark.")
    parser.add_argument(
        "--translate",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark translation (default summarize)",
    )
    parser.add_argument(
        "--save-plot", type=str, default=None, help="Where to save the bench plot"
    )

    args = parser.parse_args()

    main(args.url, args.translate, args.save_plot)
