from functools import wraps
from typing import Dict

import jwt
from config import AppConfig
from flask import request

PREFIX = "Bearer "


def verify_token(id_token: str, openid_configs: Dict):
    header = jwt.get_unverified_header(id_token)
    for conf in openid_configs:
        jwks_client = jwt.PyJWKClient(conf["jwks_uri"])

        try:
            # try matching the user token with the current jwks
            signing_key = jwks_client.get_signing_key(header["kid"])
        except Exception as e:
            # try next jwks
            continue
        else:
            # decode and verify user token
            data = jwt.decode(
                id_token,
                key=signing_key.key,
                algorithms=header["alg"],
                audience=conf["client_id"],
            )
            # return decoded token
            return data

    raise jwt.PyJWKClientError("Invalid token: Unable to find a matching signing key")


def check_token():
    def decorator(f):
        auth_config = AppConfig.get_auth_config()
        if not auth_config:
            return f

        @wraps(f)
        async def wrapper(*args, **kwargs):
            if "Authorization" in request.headers:
                bearer_token = request.headers["Authorization"]
                if not bearer_token.startswith(PREFIX):
                    return ("Error: Bearer not found", 401)
                id_token = bearer_token[len(PREFIX) :]
                try:
                    _ = verify_token(id_token, auth_config["openid_configs"])
                    return await f(*args, **kwargs)
                except Exception as e:
                    return (f"Error: {e}", 401)

            return ("Error: Missing bearer token", 401)

        return wrapper

    return decorator
