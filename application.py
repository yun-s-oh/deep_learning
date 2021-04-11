import os
from flask import Flask, flash, redirect, render_template, request, session, url_for, after_this_request
from flask_session import Session
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.utils import secure_filename
from tempfile import mkdtemp
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Configure application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):             
            flash('Not supported file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filename)
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            pil_image = Image.fromarray(image)
            # Create a Pillow ImageDraw Draw instance to draw with
            draw = ImageDraw.Draw(pil_image)
            num_faces = len(face_locations)
            app.logger.info('%s faces detected', len(face_locations))
            for (top, right, bottom, left) in face_locations:
                # Draw a box around the face using the Pillow module
                draw.rectangle(((left, top), (right, bottom)), outline= 'red', width  = 2)

            pil_image.save(filename)
            app.logger.info('file is saved as %s', filename)
            return render_template("upload.html", user_image = filename, num_faces = num_faces)

def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return f"Error Message: {e.name}({e.code})"


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)