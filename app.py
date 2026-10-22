"""Smart Attendance System using Face Recognition (Flask + OpenCV + SQLite)"""
import base64, csv, io, os, sqlite3
from datetime import datetime
import cv2, numpy as np
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
DB, DATASET, MODEL = (os.path.join(BASE, p) for p in ("attendance.db", "dataset", "model.yml"))
os.makedirs(DATASET, exist_ok=True)



detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
THRESHOLD = 85  # lower = stricter match
model_ready = False


@app.errorhandler(Exception)
def on_error(e):
    import traceback; traceback.print_exc()
    return jsonify(ok=False, results=[], msg=f"Server error: {e}"), 200


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll TEXT UNIQUE NOT NULL, name TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attendance(id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL,
            UNIQUE(student_id, date));""")


def decode(data_url):
    raw = base64.b64decode(data_url.split(",")[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def faces_in(img):
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return gray, detector.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))


def train():
    """Train the LBPH model from every image in dataset/<student_id>/"""
    global model_ready
    imgs, labels = [], []
    for sid in os.listdir(DATASET):
        folder = os.path.join(DATASET, sid)
        for f in os.listdir(folder):
            im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(im); labels.append(int(sid))
    if imgs:
        recognizer.train(imgs, np.array(labels))
        recognizer.write(MODEL)
        model_ready = True


def load_model():
    global model_ready
    if os.path.exists(MODEL):
        recognizer.read(MODEL); model_ready = True


@app.route("/")
def index():
    today = datetime.now().strftime("%Y-%m-%d")
    with db() as con:
        total = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        present = con.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    return render_template("index.html", total=total, present=present, absent=total - present)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    d = request.get_json()
    name, roll, frames = d["name"].strip(), d["roll"].strip(), d["images"]
    if not name or not roll:
        return jsonify(ok=False, msg="Name and roll number are required.")
    saved = []
    for fr in frames:
        gray, faces = faces_in(decode(fr))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            saved.append(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
    if len(saved) < 10:
        return jsonify(ok=False, msg="Face not clearly detected. Improve lighting and retry.")
    try:
        with db() as con:
            sid = con.execute("INSERT INTO students(roll,name) VALUES(?,?)", (roll, name)).lastrowid
    except sqlite3.IntegrityError:
        return jsonify(ok=False, msg="Roll number already registered.")
    folder = os.path.join(DATASET, str(sid)); os.makedirs(folder, exist_ok=True)
    for i, im in enumerate(saved):
        cv2.imwrite(os.path.join(folder, f"{i}.jpg"), im)
    train()
    return jsonify(ok=True, msg=f"{name} registered with {len(saved)} face samples.")


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    if not model_ready:
        return jsonify(results=[], msg="No students registered yet. Register a student first.")
    gray, faces = faces_in(decode(request.get_json()["image"]))
    now, results = datetime.now(), []
    if not len(faces):
        return jsonify(results=[], msg="No face detected. Move closer and face the light.")
    with db() as con:
        for (x, y, w, h) in faces:
            label, dist = recognizer.predict(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
            if dist > THRESHOLD:
                results.append(dict(name="Unknown", status="unknown")); continue
            s = con.execute("SELECT * FROM students WHERE id=?", (label,)).fetchone()
            if not s: continue
            try:
                con.execute("INSERT INTO attendance(student_id,date,time) VALUES(?,?,?)",
                            (label, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
                status = "marked"
            except sqlite3.IntegrityError:
                status = "already"
            results.append(dict(name=s["name"], roll=s["roll"], status=status))
    return jsonify(results=results, msg="Face detected, checking...")


def fetch_records(date):
    with db() as con:
        return con.execute("""SELECT s.roll, s.name, a.date, a.time FROM attendance a
            JOIN students s ON s.id=a.student_id WHERE a.date=? ORDER BY a.time""", (date,)).fetchall()


@app.route("/records")
def records():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    return render_template("records.html", rows=fetch_records(date), date=date)


@app.route("/export")
def export():
    date = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["Roll No", "Name", "Date", "Time"])
    w.writerows([tuple(r) for r in fetch_records(date)])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{date}.csv"})


if __name__ == "__main__":
    print("OpenCV version:", cv2.__version__, "| cascade loaded:", not detector.empty())
    init_db(); load_model()
    app.run(debug=True)
