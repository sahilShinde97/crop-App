from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    
    # Sample data to return as JSON
    data = {
        "message": "Hello, World!",
        "status": "success",
        "user": {
            "name": "John Doe",
            "age": 30
        }
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
