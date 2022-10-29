from flask import Flask, render_template, Response, request, send_file
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

global capture, switch, rec
capture = 0
switch = 1
rec = 0

try:
    os.mkdir('./captures')
except OSError as error:
    pass

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)

IMG_FOLDER = os.path.abspath("captures")
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(['captures', "capture_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/captured')
def captured():
    time.sleep(1)
    captures = []
    for file in os.listdir(IMG_FOLDER):
        # check only text files
        if file.endswith('.png'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
            captures.append(filepath)

    latest = max(captures, key=os.path.getmtime)
    return send_file(latest, mimetype='image/png')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()


    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
