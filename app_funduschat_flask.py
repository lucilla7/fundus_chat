# app_funduschat_flask.py
# Fundus Chat Line – runnable demo app
# Demo only — not for clinical use.

from flask import Flask, request, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import math
from sklearn.metrics.pairwise import cosine_similarity
from skimage.filters import frangi

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset' # change later to 'dataset'
FEATURES_FILE = 'features.npz'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HAVE_FRANGI = True

# --------------------------------------------------
# HTML Template
# --------------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Fundus Chat Line</title>
<style>
  body { font-family: Arial; margin: 40px; }
  .disc { color: red; font-weight: bold; }
</style>
</head>
<body>
<h2>Fundus Chat Line</h2>
<p><strong>Demo only — not for clinical use.</strong></p>
<form method="POST" enctype="multipart/form-data">
  <p>Upload Fundus Image:<br><input type=file name=image required></p>
  <p>Optional Question:<br><input type=text name=question style="width:300px"></p>
  <button type=submit>Analyze</button>
</form>

{% if result %}
<hr>
<h3>Results</h3>
<p><b>Vessel Overlay:</b></p>
<img src="{{ url_for('static', filename=result.vessel_path) }}" width=300>
<p><b>Optic Disc Overlay:</b></p>
<img src="{{ url_for('static', filename=result.disc_path) }}" width=300>

<h3>Similar Images</h3>
{% for item in result.neighbors %}
  <div style="margin-bottom:10px;">
    <img src="{{ url_for('static', filename=item.path) }}" width=200><br>
    Class: {{ item.label }}<br>
    Similarity: {{ '%.3f'|format(item.score) }}
  </div>
{% endfor %}

<h3>Summary</h3>
<div style="border:1px solid #ccc; padding:10px; width:500px;">
  <p><b>Provisional Label:</b> {{ result.provisional }}</p>
  <p>Vessel Density: {{ '%.3f'|format(result.vessel_density) }}</p>
  <p>Optic Disc Area: {{ '%.3f'|format(result.disc_area) }}</p>
  <p><b>Reply:</b> {{ result.reply }}</p>
</div>
{% endif %}
</body></html>
"""

# --------------------------------------------------
# Image utilities
# --------------------------------------------------
def load_image(path):
    return cv2.imread(path)

def vessel_overlay(img):

    g = img[:,:,1]
    clahe = cv2.createCLAHE(3.0,(8,8))
    g2 = clahe.apply(g)
    bl = cv2.medianBlur(g2,5)
    thr, mask = cv2.threshold(bl,0,255,cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[mask==0] = [0,255,0]
    return overlay, mask

def disc_overlay_old(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(1.0,(9,9))
    g2 = clahe.apply(gray)
    bl = cv2.GaussianBlur(g2,(9,9),0)
    thr, mask = cv2.threshold(bl,0,255,cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[mask==0] = [255,0,0]
    return overlay, mask


def disc_overlay(img):

    # --- Red channel (optic disc strongest here)
    r = img[:,:,2]

    # --- White Top-Hat to isolate bright local blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
    tophat = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, kernel)

    # --- Normalize
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)

    # --- Smooth
    bl = cv2.GaussianBlur(tophat, (15,15), 0)

    # --- Threshold
    _, th = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Morphological cleanup
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25)))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))

    # --- Largest component = optic disc
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    else:
        mask = th

    # --- Overlay
    overlay = img.copy()
    overlay[mask == 0] = [0, 0, 255]

    return overlay, mask


# ---------------------------------------------------------
# Feature extraction for similarity
# ---------------------------------------------------------

def hist_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256]).flatten()
    hist = hist / (np.linalg.norm(hist)+1e-8)
    return hist

# --------------------------------------------------
# Preload dataset features
# --------------------------------------------------
if os.path.exists(FEATURES_FILE):
    data = np.load(FEATURES_FILE, allow_pickle=True)
    IMAGE_PATHS = data['paths'].tolist()
    FEATURES = data['features']
    LABELS = data['labels'].tolist()
else:
    IMAGE_PATHS, FEATURES, LABELS = [], [], []
    for cls in os.listdir(DATASET_FOLDER):
        cdir = os.path.join(DATASET_FOLDER, cls)
#        if not os.path.isdir(cdir): continue
#        for f in os.listdir(cdir):
#            if not f.lower().endswith(('jpg','png','jpeg')): continue
#            p = os.path.join(cdir, f)
        if not cdir.lower().endswith(('jpg','png','jpeg')): continue
        img = load_image(cdir)
        if img is None: continue
        IMAGE_PATHS.append(cdir)
        FEATURES.append(hist_feature(img))
        LABELS.append(cls)

    FEATURES = np.array(FEATURES)
    np.savez(FEATURES_FILE, paths=IMAGE_PATHS, features=FEATURES, labels=LABELS)

# --------------------------------------------------
# Flask Route
# --------------------------------------------------
@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template_string(HTML)

    file = request.files.get('image')
    q = request.form.get('question','')

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = load_image(path)

    # Overlays
    v_overlay, vmask = vessel_overlay(img)
    d_overlay, dmask = disc_overlay(img)

    # creating path
    base = os.path.basename(path)         # currimage.jpg
    name, ext = os.path.splitext(base)    # currimage, .jpg

    v_filename = f"{name}_vessel{ext}"
    d_filename = f"{name}_disc{ext}"

    vpath = f"uploads/{v_filename}"        # relative path for HTML
    dpath = f"uploads/{d_filename}"
    
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, v_filename), v_overlay)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, d_filename), d_overlay)

    # Similarity
    qfeat = hist_feature(img).reshape(1,-1)
    sims = cosine_similarity(qfeat, FEATURES)[0]
    sims = sims[sims<1]     # remove same image
    idx = sims.argsort()[-3:][::-1]

    neighbors = []
    for i in idx:

        original = IMAGE_PATHS[i]
        # Convert dataset/... → uploads/...  (must copy file IRL)
        rel_path = original.replace("dataset/", "uploads/")
        neighbors.append({
            'path': rel_path,
            'label': LABELS[i],
            'score': float(sims[i])
        })

    # Summary
    provisional = max([n['label'] for n in neighbors], key=[n['label'] for n in neighbors].count)
    vessel_density = vmask.mean()/255
    disc_area = dmask.mean()/255

    reply = (
        f"Based on the overlays, the vessels (green) and optic disc (red) "
        f"are highlighted. The retrieved similar images helped estimate the "
        f"label '{provisional}'. Metrics computed are vessel density and disc area."
    )
    if q.strip():
        reply += " Regarding your question: this demo references the overlays and nearest images to give a simple answer."

    result_dict={
        'vessel_path': vpath,
        'disc_path': dpath,
        'neighbors': neighbors,
        'provisional': provisional,
        'vessel_density': vessel_density,
        'disc_area': disc_area,
        'reply': reply
    }

    return render_template_string(HTML, result=result_dict)


if __name__ == '__main__':
    app.run(debug=True)
