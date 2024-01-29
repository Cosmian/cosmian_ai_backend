import os
from functools import wraps

from flask import request
from google.auth.transport import requests
from google.oauth2 import id_token

PREFIX = "Bearer "
AUTH_IP = os.getenv("AUTH_IP")


def check_token():
    def decorator(f):
        if AUTH_IP is None:
            return f

        @wraps(f)
        async def wrapper(*args, **kwargs):
            if "Authorization" in request.headers:
                bearer_token = request.headers["Authorization"]
                if not bearer_token.startswith(PREFIX):
                    return ("Error: Bearer not found!", 401)
                user_token = bearer_token[len(PREFIX) :]
                try:
                    id_info = id_token.verify_oauth2_token(
                        user_token, requests.Request()
                    )
                except Exception as e:
                    # return (str(e), 401)
                    pass

                # print("User email:", id_info["email"])
                return await f(*args, **kwargs)

            return ("Invalid token!", 401)

        return wrapper

    return decorator
