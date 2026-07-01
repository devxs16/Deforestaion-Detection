from flask import Flask, render_template, request
import os

from src.change_detection import detect_changes
from src.visualize_changes import generate_change_map

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    image1 = request.files["image1"]
    image2 = request.files["image2"]

    image1_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "2018.jpg"
    )

    image2_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "2026.jpg"
    )

    image1.save(image1_path)
    image2.save(image2_path)

    changes = detect_changes(
        image1_path,
        image2_path
    )

    output_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        "change_map.png"
    )

    generate_change_map(
        image2_path,
        changes,
        output_path
    )

    return render_template(
        "result.html",
        result_image="outputs/change_map.png",
        total_changes=len(changes)
    )


if __name__ == "__main__":
    app.run(debug=True)