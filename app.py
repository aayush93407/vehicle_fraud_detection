import os
import cv2
import torch
import requests
import re
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MISTRAL_API_KEY = "aKFEMuDwJOvtphHDDOrh2qbfRP7jEA1L"

model = YOLO("best.pt")

def detect_and_annotate(path, output_path):
    ext = os.path.splitext(path)[-1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov']

    detected_types = set()
    damage_counts = {}

    if is_video:
        cap = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 640))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            results = model(frame, conf=0.25)

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    label = model.names[cls]
                    if label not in damage_counts:
                        damage_counts[label] = 0
                    damage_counts[label] += 1
                    detected_types.add(label)

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf)
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (xyxy[0], xyxy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 5)

        cap.release()
        out.release()
    else:
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 640))
        results = model(img, conf=0.25)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                label = model.names[cls]
                if label not in damage_counts:
                    damage_counts[label] = 0
                damage_counts[label] += 1
                detected_types.add(label)

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (xyxy[0], xyxy[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(output_path, img)

    return damage_counts

def markdown_table_to_html(markdown):
    rows = [row.strip() for row in markdown.strip().split('\n') if row.strip().startswith('|')]
    if len(rows) < 2:
        return None

    header = rows[0].split('|')[1:-1]
    html = '<table class="table table-bordered table-dark table-striped mt-4">\n<thead><tr>'
    for h in header:
        html += f'<th>{h.strip()}</th>'
    html += '</tr></thead>\n<tbody>'

    for row in rows[2:]:
        if '|' not in row:
            continue
        cols = row.split('|')[1:-1]
        if len(cols) != len(header):
            continue
        html += '<tr>'
        for col in cols:
            html += f'<td>{col.strip()}</td>'
        html += '</tr>'
    html += '</tbody>\n</table>'
    return html


def predict_repair_cost(car_model, damage_counts):
    if not damage_counts:
        return "No damages detected. No repair cost needed.", ""

    MODEL_NAME = "mistral-large-latest"
    unique_damages = list(damage_counts.keys())
    damage_description = "\n".join([f"- {d}" for d in unique_damages])
    prompt_text = (
        f"Car Model: {car_model}\n"
        f"Detected damages (only once per type):\n{damage_description}\n\n"
        "Provide a detailed repair cost breakdown for each damage type with average market estimates in India. And at the end, "
        "provide a total cost estimate. "
        "At the end, include a markdown table titled 'Repair Cost Breakdown Table' with columns for Damage Type, Parts Cost (INR), and Labor Cost (INR). "
        "Use pipe (`|`) separators like a proper markdown table."
    )

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt_text}], "max_tokens": 800},
            timeout=120
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Look for a markdown table (find any section that looks like a table)
            table_lines = []
            start_collecting = False
            for line in content.splitlines():
                if "Repair Cost Breakdown Table" in line:
                    start_collecting = True
                elif start_collecting and line.strip().startswith("|"):
                    table_lines.append(line)
                elif start_collecting and not line.strip().startswith("|"):
                    break  # stop when table ends

            markdown_table = "\n".join(table_lines)
            table_html = markdown_table_to_html(markdown_table) if markdown_table else ""

            return content, table_html
        else:
            return f"API Error: {response.status_code} {response.text}", ""
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}", ""


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        car_model = request.form['car_model']
        file = request.files['media']
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        output_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
        damage_counts = detect_and_annotate(input_path, output_path)
        repair_cost, repair_table_html = predict_repair_cost(car_model, damage_counts)

        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
        processed_media = f"processed_{filename}" if os.path.exists(output_path) else None

        return render_template("result.html", model=car_model, damages=damage_counts,
                               cost=repair_cost, repair_table=repair_table_html,
                               processed_media=processed_media, is_video=is_video)

    return render_template("upload.html")

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)