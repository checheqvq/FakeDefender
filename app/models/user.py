from flask_login import UserMixin


class User(UserMixin):
    """用户类"""

    def __init__(self, phone):
        self.phone = phone

    def __repr__(self):
        return "<User (%s)>" % self.phone

    def get_id(self):
        return self.phone
