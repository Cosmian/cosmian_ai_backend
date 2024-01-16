import asyncio

from app import app_asgi
from hypercorn.asyncio import serve
from hypercorn.config import Config


def main():
    config_map = {
        "bind": "0.0.0.0:5000",
        "alpn_protocols": ["h2"],
        "workers": 1,
        "accesslog": "-",
        "errorlog": "-",
        "worker_class": "uvloop",
        "wsgi_max_body_size": 2 * 1024 * 1024 * 1024,  # 2 GB
    }

    config = Config.from_mapping(config_map)
    asyncio.run(serve(app_asgi, config))


if __name__ == "__main__":
    main()
