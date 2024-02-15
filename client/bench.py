import threading
import time

import matplotlib.pyplot as plt
from summarize import summarize_data
from translate import translate_data


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
    content = open("./data/doc.txt", "rb").read()

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
    content = open("./data/doc.txt", "rb").read()

    print(f"Sending {nb_queries} queries of {len(content)} bytes")

    requests = []
    for _ in range(nb_queries):
        t = threading.Thread(target=summarize_data, args=(content, url))
        t.start()
        requests.append(t)

    for t in requests:
        t.join()


if __name__ == "__main__":
    # url = "https://demo-vm.staging.cosmian.com"
    url = "http://localhost:5001"

    nb_requests = list(range(1, 10, 2))
    durations = []
    for i in nb_requests:
        durations.append(bench_summarize(url, i))

    plt.scatter(nb_requests, durations)
    plt.xlabel("Number of requests")
    plt.ylabel("Response time")
    plt.title("Summarize")
    plt.savefig("bench_plot/amx/summarize_avx512.jpg")

