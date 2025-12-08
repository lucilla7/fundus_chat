
from flask import Flask, request, render_template_string, send_from_directory, url_for
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Folders
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
DATASET_FOLDER = 'dataset'
FEATURES_FILE = 'features.npz'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# -------------------
# HTML Template
# -------------------
HTML = r"""
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
  <p>Upload Fundus Image:<br><input type="file" name="image" required></p>
  <p>Optional Question:<br><input type="text" name="question" style="width:300px"></p>
  <button type="submit">Analyze</button>
</form>

{% if result %}
<hr>
<h3>Results</h3>
<p><b>Vessel Overlay:</b></p>
<img src="{{ url_for('static',verlay:</b></p>
{{ url_for(

<h3>Similar Images</h3>
{% for item in result.neighbors %}
  <div style="margin-bottom:10px;">
    {{ url_for(<br>
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
</body>
</html>
"""

# -------------------
# Image utilities
# -------------------
def load_image(path):
    return cv2.imread(path)

def vessel_overlay(img):
    g = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g2 = clahe.apply(g)
    bl = cv2.medianBlur(g2, 5)
    _, mask = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[mask == 0] = [0, 255, 0]
    return overlay, mask

def disc_overlay(img):
    r = img[:, :, 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    tophat = cv2.morphologyEx(r, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    bl = cv2.GaussianBlur(tophat, (15, 15), 0)
    _, th = cv2.threshold(bl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    else:
        mask = th
    overlay = img.copy()
    overlay[mask == 0] = [0, 0, 255]
    return overlay, mask

# -------------------
# Features
# -------------------
def hist_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (np.linalg.norm(hist) + 1e-8)
    return hist

# -------------------
# Preload dataset features
# -------------------
def build_dataset_features():
    image_paths = []
    labels = []
    feats = []

    if not os.path.isdir(DATASET_FOLDER):
        return [], [], np.empty((0, 512), dtype=np.float32)

    for cls in sorted(os.listdir(DATASET_FOLDER)):
        class_dir = os.path.join(DATASET_FOLDER, cls)
        if not os.path.isdir(class_dir):
            # also allow flat datasets (images directly in DATASET_FOLDER)
            if cls.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img = load_image(class_dir)
                if img is not None:
                    image_paths.append(class_dir)
                    labels.append('unknown')
                    feats.append(hist_feature(img))
            continue

        for f in sorted(os.listdir(class_dir)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                continue
            p = os.path.join(class_dir, f)
            img = load_image(p)
            if img is None:
                continue
            image_paths.append(p)
            labels.append(os.path.basename(class_dir))
            feats.append(hist_feature(img))

    if feats:
        feats = np.vstack(feats)
    else:
        feats = np.empty((0, 512), dtype=np.float32)

    return image_paths, labels, feats

if os.path.exists(FEATURES_FILE):
    data = np.load(FEATURES_FILE, allow_pickle=True)
    IMAGE_PATHS = data['paths'].tolist()
    LABELS = data['labels'].tolist()
    FEATURES = data['features']
else:
    IMAGE_PATHS, LABELS, FEATURES = build_dataset_features()
    np.savez(FEATURES_FILE, paths=np.array(IMAGE_PATHS, dtype=object),
             labels=np.array(LABELS, dtype=object), features=FEATURES)

# -------------------
# Routes
# -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(HTML)

    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template_string(HTML, result=None)

    q = request.form.get('question', '').strip()

    # Save upload
    filename = os.path.basename(file.filename)
    safe_name = filename.replace(' ', '_')
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(path)

    img = load_image(path)
    if img is None:
        return render_template_string(HTML, result=None)

    # Overlays
    v_overlay, vmask = vessel_overlay(img)
    d_overlay, dmask = disc_overlay(img)

    name, ext = os.path.splitext(safe_name)
    v_filename = f"{name}_vessel{ext}"
    d_filename = f"{name}_disc{ext}"
    vpath = os.path.join('uploads', v_filename).replace('\\', '/')
    dpath = os.path.join('uploads', d_filename).replace('\\', '/')
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, v_filename), v_overlay)
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, d_filename), d_overlay)

    # Similarity (keep original indices)
    neighbors = []
    if FEATURES is not None and len(FEATURES) > 0:
        qfeat = hist_feature(img).reshape(1, -1)
        sims_full = cosine_similarity(qfeat, FEATURES)[0]

        # Exclude identical image by filename if dataset contains the same file
        # (optional: threshold exclude 0.999+)
        topk_idx = np.argsort(sims_full)[-3:][::-1]
        for i in topk_idx:
            original = IMAGE_PATHS[i]
            # Serve dataset images under static by computing a relative path from static/
            # Option A: expose dataset under static/dataset (symlink or ensure path)
            # Here we construct a URL path relative to static folder:
            # If DATASET_FOLDER is 'dataset' at project root, place a symlink under static/ if needed.
            rel_under_static = os.path.join('..', original).replace('\\', '/')
            # Simpler: copy the file into uploads (commented out for brevity)
            # shutil.copy2(original, os.path.join(UPLOAD_FOLDER, os.path.basename(original)))
            # rel_under_static = os.path.join('uploads', os.path.basename(original))

            neighbors.append({
                'path': rel_under_static,  # may require static serving setup
                'label': LABELS[i],
                'score': float(sims_full[i])
            })

    # Provisional label by simple majority over neighbors
    if neighbors:
        labels_list = [n['label'] for n in neighbors]
        provisional = max(labels_list, key=labels_list.count)
    else:
        provisional = 'unknown'

    vessel_density = float(vmask.mean() / 255.0)
    disc_area = float(dmask.mean() / 255.0)

    reply = (
        f"Based on the overlays, the vessels (green) and optic disc (red) are highlighted. "
        f"The retrieved similar images helped estimate the label '{provisional}'. "
        f"Metrics computed are vessel density and disc area."
    )
    if q:
        reply += " Regarding your question: this demo references the overlays and nearest images to give a simple answer."

    result_dict = {
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
    # For local testing only
    app.run(host='0.0.0.0', port=5000, debug=True)
