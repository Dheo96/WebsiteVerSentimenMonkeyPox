document.getElementById("sentiment-form").addEventListener("submit", function (e) {
  e.preventDefault();
  const tweet = document.getElementById("tweet").value;

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: tweet }),
  })
    .then((response) => response.json())
    .then((data) => {
      const predictionText = document.getElementById("prediction-text");
      const prediction = data.prediction || "Error";

      predictionText.textContent = prediction.toUpperCase();
      predictionText.className = "";

      if (prediction === "positive") {
        predictionText.classList.add("sentiment-positive");
      } else if (prediction === "negative") {
        predictionText.classList.add("sentiment-negative");
      } else if (prediction === "neutral") {
        predictionText.classList.add("sentiment-neutral");
      } else {
        predictionText.classList.add("sentiment-error");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      const predictionText = document.getElementById("prediction-text");
      predictionText.textContent = "Error";
      predictionText.className = "sentiment-error";
    });
});
