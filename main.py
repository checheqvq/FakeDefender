from app import app


@app.route('/', methods=['POST', 'GET'])
def hello():
    print("test")
    return "Welcome to Deepfake Server."


if __name__ == "__main__":
    app.run(
        host='127.0.0.1',
        port=5000
    )
