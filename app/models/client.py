from werkzeug.security import generate_password_hash, check_password_hash

from ..extensions import db


# get an instance from client database
def get_client(username=None):
    client = None
    if username:
        if username == "admin":
            client = Client(username, password="password")
    return client


def add_client(username, password):
    new_client = Client(username=username, password=generate_password_hash(password))
    db.session.add(new_client)
    db.session.commit()


class Client:
    def __init__(self, username, password):
        self.cID = 0
        self.username = username
        self.password = password

    def __repr__(self):
        return "<Client (%s)>" % self.cID

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return self.cID
