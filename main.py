import base64
import datetime
import os, shutil
import numpy as np
import tensorflow as tf
from models.functions import get_image_array, visualize_segmentation, class_colors
from pathlib import Path
from flask import Flask, render_template, request
from PIL import Image
from keras import backend as K
from models.fpnresnet50 import fpn_8_resnet50

os.system("pip3 install --upgrade pip")
os.system("pip3 install opencv-python")

import cv2

try:
    os.mkdir('./captures')
except OSError as error:
    pass

try:
    os.mkdir('./segments')
except OSError as error:
    pass

try:
    os.mkdir('./masks')
except OSError as error:
    pass
# instantiate flask app
app = Flask(__name__, template_folder='./templates')

IMG_FOLDER = os.path.abspath("captures")
OUT_FOLDER = os.path.abspath("segments")
MASK_FOLDER = os.path.abspath("masks")
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

    for filename in os.listdir(MASK_FOLDER):
        file_path = os.path.join(MASK_FOLDER, filename)
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
    for imageFile in os.listdir(IMG_FOLDER):
        # check only image files
        if imageFile.endswith('.png'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], imageFile)
            captures.append(filepath)

    latest = max(captures, key=os.path.getmtime)

    path = Path("./fcn_8_resnet50.h5")
    model = fpn_8_resnet50()
    model.load_weights(path)

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

    mask_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                     colors=class_colors, overlay_img=False,
                                     prediction_width=output_width,
                                     prediction_height=output_height)

    out = Image.fromarray(seg_img.astype("uint8"))
    outMask = Image.fromarray(mask_img.astype("uint8"))
    K.clear_session()

    out.save("./segments/output"+format(str(now).replace(":", ''))+".png", "PNG")
    outMask.save("./masks/mask"+format(str(now).replace(":", ''))+".png", "PNG")
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


@app.route('/mask', methods=['POST'])
def toggle_mask():
    segments = []
    masks = []
    toggle = request.form["toggle"]
    if toggle == "1":
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

    elif toggle == "0":
        for file in os.listdir(MASK_FOLDER):
            # check only text files
            if file.endswith('.png'):
                filepath = os.path.join(MASK_FOLDER, file)
                masks.append(filepath)
        out_mask = max(masks, key=os.path.getmtime)
        with open(out_mask, "rb") as img_file:
            img64 = base64.b64encode(img_file.read())
        base64string = 'data:image/png;base64,' + img64.decode('utf-8')
        return base64string


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, ssl_context='adhoc')
