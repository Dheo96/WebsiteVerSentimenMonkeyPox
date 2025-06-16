from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Muat model dan vectorizer
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"prediction": "Error"}), 400
        
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"prediction": "Error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
