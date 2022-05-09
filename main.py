from distutils.log import debug
from app import app


@app.route('/', methods=['POST', 'GET'])
def hello():
    print("test")
    return "Welcome to Deepfake Server."


if __name__ == "__main__":
    app.run(
        host='10.136.126.13',
        port=5000
    )
