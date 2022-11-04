import base64
import datetime
import os, shutil
import numpy as np
from models.functions import get_image_array, visualize_segmentation, class_colors
from pathlib import Path
from flask import Flask, render_template, request
from fcn8 import fcn_8
from PIL import Image
import cv2

try:
    os.mkdir('./captures')
except OSError as error:
    pass

try:
    os.mkdir('./segments')
except OSError as error:
    pass

# instantiate flask app
app = Flask(__name__, template_folder='./templates')

IMG_FOLDER = os.path.abspath("captures")
OUT_FOLDER = os.path.abspath("segments")
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/')
def index():
    for filename in os.listdir(IMG_FOLDER):
        file_path = os.path.join(IMG_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception:
            pass

    for filename in os.listdir(OUT_FOLDER):
        file_path = os.path.join(OUT_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass

    return render_template('index.html')


@app.route('/captured', methods=['POST'])
def captured():

    file = request.form["image"]
    imgdata = base64.b64decode(file)
    now = datetime.datetime.now()
    p = os.path.sep.join(['captures', "shot_{}.png".format(str(now).replace(":", ''))])
    with open(p, 'wb') as f:
        f.write(imgdata)

    captures = []
    for file in os.listdir(IMG_FOLDER):
        # check only text files
        if file.endswith('.png'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
            captures.append(filepath)

    latest = max(captures, key=os.path.getmtime)

    path = Path('fcn_8_resnet50.00005')
    model = fcn_8(51, input_height=480, input_width=640, channels=3)
    #model.load_weights(path)

    input_width = model.input_width
    input_height = model.input_height
    output_width = model.output_width
    output_height = model.output_height
    n_classes = model.n_classes

    x = get_image_array(latest, input_width, input_height)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    inp = cv2.imread(latest, 1)

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=class_colors, overlay_img=True,
                                     prediction_width=output_width,
                                     prediction_height=output_height)

    out = Image.fromarray(seg_img.astype("uint8"))
    out.save("./segments/output"+format(str(now).replace(":", ''))+".png", "PNG")
    segments = []
    for file in os.listdir(OUT_FOLDER):
        # check only text files
        if file.endswith('.png'):
            filepath = os.path.join(OUT_FOLDER, file)
            segments.append(filepath)
    out_seg = max(segments, key=os.path.getmtime)
    with open(out_seg, "rb") as img_file:
        img64 = base64.b64encode(img_file.read())
    base64string = 'data:image/png;base64,' + img64.decode('utf-8')
    return base64string


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, ssl_context='adhoc')
