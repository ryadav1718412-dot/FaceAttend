# Smart Attendance System using Face Recognition
**Stack:** Python (Flask), OpenCV (Haar Cascade + LBPH), SQLite, HTML/CSS/JS

## Run
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 (camera works on localhost).

## Flow
1. Register Student -> 30 face samples captured, model trained.
2. Take Attendance -> camera recognises faces, marks present once per day.
3. Records -> filter by date, download CSV.

## Structure
app.py (backend) | templates/ (HTML) | static/style.css | dataset/ (face images) | attendance.db
