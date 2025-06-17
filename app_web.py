import os
import pandas as pd
from flask import Flask, render_template, request, send_file
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

label_map = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

@app.route("/", methods=["GET", "POST"])
def home():
    tweet_sentiment = None
    csv_summary = None
    preview_data = None
    chart_path = None

    if request.method == "POST":
        if "tweet" in request.form:
            tweet = request.form["tweet"]
            vector = vectorizer.transform([tweet])
            prediction = model.predict(vector)[0]
            tweet_sentiment = label_map.get(prediction, prediction)
            return render_template("index.html", tweet=tweet, sentiment=tweet_sentiment)

        elif "file" in request.files:
            file = request.files["file"]
            if not file:
                return "Tidak ada file CSV", 400

            try:
                df = pd.read_csv(file)

                if "tweet" not in df.columns:
                    return "CSV harus punya kolom 'tweet'", 400

                texts = df["tweet"].astype(str).tolist()
                vectors = vectorizer.transform(texts)
                predictions = model.predict(vectors)

                df["Sentimen"] = [label_map.get(p, p) for p in predictions]
                df.to_csv("static/output.csv", index=False)

                csv_summary = df["Sentimen"].value_counts().sort_index()
                preview_data = df[["tweet", "Sentimen"]].head(10).values.tolist()

                # Grafik Batang
                plt.figure(figsize=(8, 5))
                sns.set_style("whitegrid")
                sns.barplot(x=csv_summary.index, y=csv_summary.values, palette="Set2")
                plt.title("Distribusi Sentimen")
                plt.xlabel("Sentimen")
                plt.ylabel("Jumlah")
                chart_path = "static/chart.png"
                plt.savefig(chart_path)
                plt.close()

                return render_template("index.html",
                                       tweet=tweet_sentiment,
                                       csv_summary=csv_summary.to_dict(),
                                       preview=preview_data,
                                       chart=chart_path)

            except Exception as e:
                return f"Terjadi error saat proses: {str(e)}", 500

    return render_template("index.html")

@app.route("/download_result")
def download_result():
    return send_file("static/output.csv", as_attachment=True, download_name="hasil_sentimen.csv")

if __name__ == "__main__":
    app.run(debug=True)
