from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


# 🔥 LOAD MODEL (RUNS ONLY ONCE)
proto = "model/colorization_deploy_v2.prototxt"
model = "model/colorization_release_v2.caffemodel"
points = "model/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(proto, model)
pts = np.load(points)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


# 🎨 COLORIZATION FUNCTION (REAL AI)
def colorize_image(input_path, output_path):
    image = cv2.imread(input_path)

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = resized[:, :, 0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = lab[:, :, 0]

    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    cv2.imwrite(output_path, colorized)


# 🖥️ UI ROUTE
@app.route("/")
def index():
    return render_template("index.html")


# ⚡ API ROUTE
@app.route("/colorize", methods=["POST"])
def colorize():
    try:
        file = request.files["image"]

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)

        file.save(input_path)

        colorize_image(input_path, output_path)

        return jsonify({
            "input_image": input_path,
            "output_image": output_path
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)