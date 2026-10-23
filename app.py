from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        if message.strip():

            # Convert text into Bag-of-Words
            message_vector = vectorizer.transform([message])

            # Predict class
            prediction = model.predict(message_vector)[0]

            # Get confidence
            probabilities = model.predict_proba(message_vector)
            confidence = round(max(probabilities[0]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)