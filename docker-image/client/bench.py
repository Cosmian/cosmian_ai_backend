import threading
import time

from summarize import summarize_data
from translate import translate_data


def timed(func):
    """
    records approximate durations of function calls
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        print("{name:<30} started".format(name=func.__name__))
        result = func(*args, **kwargs)
        duration = "{name:<30} finished in {elapsed:.2f} seconds".format(
            name=func.__name__, elapsed=time.time() - start
        )
        print(duration)
        return result

    return wrapper


@timed
def bench_translate(nb_queries=10):
    content = open("doc.txt", "rb").read()
    url = "http://127.0.0.1:5000"

    print(f"Sending {nb_queries} queries of {len(content)} bytes")

    requests = []
    for _ in range(nb_queries):
        t = threading.Thread(target=translate_data, args=(content, url))
        t.start()
        requests.append(t)

    for t in requests:
        t.join()


@timed
def bench_summarize(nb_queries=10):
    content = open("doc.txt", "rb").read()
    url = "http://127.0.0.1:5000"

    print(f"Sending {nb_queries} queries of {len(content)} bytes")

    requests = []
    for _ in range(nb_queries):
        t = threading.Thread(target=summarize_data, args=(content, url))
        t.start()
        requests.append(t)

    for t in requests:
        t.join()


if __name__ == "__main__":
    bench_translate(5)
    bench_summarize(10)
