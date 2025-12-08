# app_funduschat_flask.py
# Fundus Chat Line – runnable demo app
# Demo only — not for clinical use.

from flask import Flask, request, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import torch
from unet import UNetSmall

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset' # change later to 'dataset'
FEATURES_FILE = 'features.npz'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
# Models
# --------------------------------------------------
MODEL_VESSEL = 'models/model_vessels.pth'
MODEL_DISC = 'models/model_disc.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------
# Image utilities
# --------------------------------------------------
def load_image(path):
    return cv2.imread(path)


# ---------- Fast fallback methods ----------
def fast_vessels(img):
    g = img[:,:,1]
    clahe = cv2.createCLAHE(2.0,(8,8))
    g2 = clahe.apply(g)
    bl = cv2.medianBlur(g2,5)
    thr, mask = cv2.threshold(bl,0,255,cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[mask==0] = [0,255,0]
    return overlay, mask

def fast_disc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8))
    g2 = clahe.apply(gray)
    bl = cv2.GaussianBlur(g2,(9,9),0)
    thr, mask = cv2.threshold(bl,0,255,cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[mask==0] = [255,0,0]
    return overlay, mask

# ---------------------------------------------------------
# CNN loading
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vessel_model = None
disc_model  = None

if os.path.exists("models/model_vessels.pth"):
    vessel_model = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    vessel_model.load_state_dict(torch.load("models/model_vessels.pth", map_location=device))
    vessel_model.eval()

if os.path.exists("models/model_disc.pth"):
    disc_model = UNetSmall(in_ch=3, out_ch=1, base=32).to(device)
    disc_model.load_state_dict(torch.load("models/model_disc.pth", map_location=device))
    disc_model.eval()

# ---------------------------------------------------------
# CNN inference
# ---------------------------------------------------------
def cnn_infer(model, img_bgr, thr=0.4):
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512,512))
    t = img_resized.astype("float32") / 255.0
    t = torch.from_numpy(t.transpose(2,0,1)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t)[0,0].cpu().numpy()

    pred = cv2.resize(pred, (W,H))
    mask = (pred >= thr).astype("uint8") * 255
    return mask

# ---------------------------------------------------------
# HTML image overlay generation
# ---------------------------------------------------------
def vessel_overlay(img):
    """Try CNN, fallback to fast."""
    if vessel_model is not None:
        mask = cnn_infer(vessel_model, img)
        overlay = img.copy()
        overlay[mask>0] = (0,255,0)
        return mask, overlay
    else:
        return fast_vessels(img)

def disc_overlay(img):
    if disc_model is not None:
        mask = cnn_infer(disc_model, img)
        overlay = img.copy()
        overlay[mask>0] = (0,0,255)
        return mask, overlay
    else:
        return fast_disc(img)
    
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