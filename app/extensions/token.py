from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from ..config import SECRET_KEY

serializer = Serializer(SECRET_KEY, expires_in=60 * 60 * 24)


def gen_token(address):
    return serializer.dumps(address).decode()


def verify_token(token):
    try:
        email = serializer.loads(token)
        return email
    except Exception:
        return None
