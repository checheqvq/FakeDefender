from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from ..extensions import db


def get_user(phone=None):
    user = None
    if phone:
        user = User.query.filter_by(phone=phone).first()
    return user


def add_user(phone, password):
    new_user = User(phone=phone, password=generate_password_hash(password))
    db.session.add(new_user)
    db.session.commit()


def confirm_user(phone):
    db.session.query(User).filter_by(phone=phone).update({User.is_confirmed: 1})
    db.session.commit()


class User(db.Model, UserMixin):
    """用户类"""

    __tablename__ = 'user'

    phone = db.Column(db.String(50), primary_key=True, unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=True)
    is_confirmed = db.Column(db.Boolean(1), default=0)

    def __repr__(self):
        return "<User (%s)>" % self.phone

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return self.phone
