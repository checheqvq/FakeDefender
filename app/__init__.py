from flask import Flask

from app.config import *
from app.extensions import *
from app.views.user import user_app


app = Flask(__name__)

app.config['SECRET_KEY'] = SECRET_KEY

app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS


app.register_blueprint(user_app, url_prefix='/user')

login_manager.init_app(app)
db.init_app(app)
logger_init_app(app)

