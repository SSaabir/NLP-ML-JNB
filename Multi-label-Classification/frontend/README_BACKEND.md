If you don't have a backend yet, a minimal Flask example (Python) for the `/predict` endpoint:

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    title = data.get('title','')
    overview = data.get('overview','')
    # call your model here and return tags
    return jsonify({'tags': ['Drama','Action']})

if __name__ == '__main__':
    app.run(port=8000)
```

Remember to enable CORS or proxy from Vite if running on different ports.
