from flask import (
    Flask,
    render_template,
    Response,
    redirect,
    request,
    jsonify,
    make_response,
    url_for
)

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import time
import os
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from urllib.parse import urlparse
import urllib.parse
import io

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image as RLImage,
    KeepInFrame
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# IMPORTANT:
# Set your existing PostgreSQL DATABASE_URL as an environment
# variable before running the application.
#
# Example:
# export DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@localhost:5432/Fab"
#
# Or configure it in your .env / terminal environment.

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("WARNING: DATABASE_URL environment variable is not set.")
    print("Database features will not work until DATABASE_URL is configured.")

db_config = None

if DATABASE_URL:
    try:
        url = urlparse(DATABASE_URL)

        db_config = {
            "dbname": url.path[1:],
            "user": url.username,
            "password": urllib.parse.unquote(url.password)
            if url.password
            else "",
            "host": url.hostname,
            "port": url.port or 5432
        }

    except Exception as e:
        print("Database URL error:", e)
        db_config = None


def get_db_connection():
    if db_config is None:
        raise RuntimeError(
            "Database is not configured. "
            "Set the DATABASE_URL environment variable."
        )

    return psycopg2.connect(**db_config)


# ============================================================
# GLOBAL STATE
# ============================================================

active_batch = None

SAVE_DIR = "static/defects"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "best_model.h5"

# IMPORTANT:
# This order MUST exactly match the class_indices generated
# during training in Google Colab.
#
# Your trained model has:
#
# 0 = Broken stitch
# 1 = Needle mark
# 2 = Pinched fabric
# 3 = Vertical
# 4 = defect free
# 5 = hole
# 6 = horizontal
# 7 = lines
# 8 = stain

CLASS_NAMES = [
    "broken_stitch",
    "needle_mark",
    "pinched_fabric",
    "vertical",
    "defect-free",
    "hole",
    "horizontal",
    "lines",
    "stain"
]

INPUT_SIZE = (224, 224)

# The application displays DEFECT! only when
# confidence is greater than or equal to this value.

DEFECT_THRESHOLD = 0.85


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("WEAVECHECK - FABRIC DEFECT DETECTION")
print("=" * 60)

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found: {MODEL_PATH}\n\n"
        f"Place your trained model '{MODEL_PATH}' in:\n"
        f"{os.path.abspath('.')}\n"
    )

# compile=False because this application only performs inference.
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

print("Model output shape:", model.output_shape)

# Make sure the trained model really has 9 outputs.
if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Model has {model.output_shape[-1]} output classes, "
        f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
    )

print("Number of classes:", len(CLASS_NAMES))

for i, class_name in enumerate(CLASS_NAMES):
    print(f"{i}: {class_name}")

print("=" * 60)


# ============================================================
# SAVE CONTROL
# ============================================================

last_save_time = 0

MIN_SAVE_INTERVAL = 3


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img):
    """
    Convert OpenCV BGR image to RGB,
    resize to 224x224,
    normalize to 0-1,
    and add batch dimension.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)

    img = img.resize(INPUT_SIZE)

    img = np.array(img).astype(np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(frame):
    """
    Perform one prediction on a frame.

    Returns:
        predicted_class
        confidence
        probabilities
    """

    processed = preprocess_image(frame)

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_index])

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence, probabilities


# ============================================================
# TEST-TIME AUGMENTATION
# ============================================================

def predict_with_tta(frame):
    """
    Predict using:
    1. Original image
    2. Horizontal flip
    3. Vertical flip

    The predictions are averaged.
    """

    predictions = []

    # Original
    predictions.append(
        model.predict(
            preprocess_image(frame),
            verbose=0
        )[0]
    )

    # Horizontal flip
    horizontal_flip = cv2.flip(frame, 1)

    predictions.append(
        model.predict(
            preprocess_image(horizontal_flip),
            verbose=0
        )[0]
    )

    # Vertical flip
    vertical_flip = cv2.flip(frame, 0)

    predictions.append(
        model.predict(
            preprocess_image(vertical_flip),
            verbose=0
        )[0]
    )

    # Average probabilities
    avg_prediction = np.mean(
        predictions,
        axis=0
    )

    predicted_index = int(
        np.argmax(avg_prediction)
    )

    confidence = float(
        avg_prediction[predicted_index]
    )

    predicted_class = CLASS_NAMES[predicted_index]

    return (
        predicted_class,
        confidence,
        avg_prediction
    )


# ============================================================
# DATABASE - SAVE DEFECT
# ============================================================

def save_defect(
    defect_name,
    confidence,
    image_path
):

    global last_save_time
    global active_batch

    # Avoid saving the same defect continuously
    if time.time() - last_save_time < MIN_SAVE_INTERVAL:
        return False

    # Batch must be active
    if not active_batch:
        print("No active batch -> defect not saved")
        return False

    try:

        conn = get_db_connection()

        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO defects
            (
                defect_name,
                confidence,
                image_path,
                batch_id,
                detected_at
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                NOW()
            )
            """,
            (
                defect_name,
                float(confidence),
                image_path,
                active_batch["batch_id"]
            )
        )

        conn.commit()

        cur.close()
        conn.close()

        last_save_time = time.time()

        print(
            f"[SAVED] "
            f"Batch #{active_batch['batch_id']} | "
            f"{defect_name} "
            f"({confidence:.1%})"
        )

        return True

    except Exception as e:

        print("DB Error:", e)

        return False


# ============================================================
# CAMERA STREAM
# ============================================================

def generate_frames():

    # Mac-compatible camera initialization.
    #
    # Camera 0 = built-in/default camera.
    #
    # If you use an external USB camera, try 1 instead.

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Could not open camera.")

        placeholder_path = "static/no_camera.jpg"

        if os.path.exists(placeholder_path):

            placeholder = open(
                placeholder_path,
                "rb"
            ).read()

            while True:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + placeholder
                    + b"\r\n"
                )

                time.sleep(0.1)

        return

    print("Camera started.")

    while True:

        ret, frame = cap.read()

        if not ret:

            print("Failed to read camera frame.")

            break

        # ====================================================
        # PREDICTION
        # ====================================================

        pred_class, confidence, probabilities = (
            predict_with_tta(frame)
        )

        # ====================================================
        # DISPLAY CLASS NAME
        # ====================================================

        display_class = (
            pred_class
            .replace("_", " ")
            .title()
        )

        confidence_percent = confidence * 100

        label = (
            f"{display_class} "
            f"({confidence_percent:.1f}%)"
        )

        # ====================================================
        # COLOR
        # ====================================================

        if pred_class == "defect-free":

            color = (0, 255, 0)

        else:

            color = (0, 0, 255)

        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            label,
            (20, 60),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            color,
            3
        )

        # ====================================================
        # DEFECT ALERT
        # ====================================================

        if (
            pred_class != "defect-free"
            and confidence >= DEFECT_THRESHOLD
        ):

            # Red border

            cv2.rectangle(
                frame,
                (0, 0),
                (
                    frame.shape[1] - 1,
                    frame.shape[0] - 1
                ),
                (0, 0, 255),
                20
            )

            # DEFECT text

            cv2.putText(
                frame,
                "DEFECT!",
                (
                    frame.shape[1] // 4,
                    160
                ),
                cv2.FONT_HERSHEY_DUPLEX,
                3.0,
                (0, 0, 255),
                7
            )

            # =================================================
            # SAVE DEFECT IMAGE
            # =================================================

            filename = (
                f"{pred_class}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                f".jpg"
            )

            full_path = os.path.join(
                SAVE_DIR,
                filename
            )

            cv2.imwrite(
                full_path,
                frame
            )

            web_path = (
                f"/{SAVE_DIR}/{filename}"
            )

            save_defect(
                pred_class,
                confidence,
                web_path
            )

        # ====================================================
        # ENCODE FRAME
        # ====================================================

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                85
            ]
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

    cap.release()

    print("Camera stopped.")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# START BATCH
# ============================================================

@app.route(
    "/start_batch",
    methods=["GET", "POST"]
)
def start_batch():

    global active_batch

    if request.method == "POST":

        batch_name = request.form[
            "batch_name"
        ].strip()

        fabric_type = request.form[
            "fabric_type"
        ].strip()

        try:

            conn = get_db_connection()

            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO batch_details
                (
                    batch_name,
                    fabric_type,
                    created_at,
                    status
                )
                VALUES
                (
                    %s,
                    %s,
                    NOW(),
                    'active'
                )
                RETURNING
                    batch_id,
                    batch_name,
                    fabric_type
                """,
                (
                    batch_name,
                    fabric_type
                )
            )

            result = cur.fetchone()

            conn.commit()

            cur.close()
            conn.close()

            active_batch = {
                "batch_id": result[0],
                "batch_name": result[1],
                "fabric_type": result[2]
            }

            print(
                f"New batch started: "
                f"#{active_batch['batch_id']} - "
                f"{active_batch['batch_name']}"
            )

            return redirect("/")

        except Exception as e:

            print(
                "Batch creation error:",
                e
            )

            return (
                "<h3>Error creating batch:</h3>"
                f"<p>{e}</p>"
            ), 500

    return render_template(
        "start_batch.html"
    )


# ============================================================
# FINISH BATCH
# ============================================================

@app.route("/finish_batch")
def finish_batch():

    global active_batch

    if active_batch:

        try:

            conn = get_db_connection()

            cur = conn.cursor()

            cur.execute(
                """
                UPDATE batch_details
                SET
                    status = 'completed',
                    ended_at = NOW()
                WHERE batch_id = %s
                """,
                (
                    active_batch["batch_id"],
                )
            )

            conn.commit()

            cur.close()
            conn.close()

            print(
                f"Batch #{active_batch['batch_id']} finished."
            )

            active_batch = None

        except Exception as e:

            print(
                "Finish batch error:",
                e
            )

    return redirect("/")


# ============================================================
# CURRENT BATCH
# ============================================================

@app.route("/current_batch")
def current_batch():

    if active_batch:

        return jsonify(
            {
                "id": active_batch["batch_id"],
                "name": active_batch["batch_name"],
                "fabric_type": active_batch["fabric_type"]
            }
        )

    return jsonify(
        {
            "id": None,
            "name": "No Active Batch",
            "fabric_type": ""
        }
    )


# ============================================================
# DEFECT COUNT
# ============================================================

@app.route("/defect_count")
def defect_count():

    try:

        conn = get_db_connection()

        cur = conn.cursor()

        if active_batch:

            cur.execute(
                """
                SELECT COUNT(*)
                FROM defects
                WHERE batch_id = %s
                """,
                (
                    active_batch["batch_id"],
                )
            )

        else:

            cur.execute(
                "SELECT COUNT(*) FROM defects"
            )

        count = cur.fetchone()[0]

        cur.close()
        conn.close()

        return {
            "count": count
        }

    except Exception as e:

        print(
            "Defect count error:",
            e
        )

        return {
            "count": 0
        }


# ============================================================
# DEFECT LIST
# ============================================================

@app.route("/defects")
def defects_list():

    batch_id = request.args.get(
        "batch_id"
    )

    try:

        conn = get_db_connection()

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # ====================================================
        # BATCHES
        # ====================================================

        cur.execute(
            """
            SELECT
                batch_id,
                batch_name,
                fabric_type,
                created_at,
                status
            FROM batch_details
            ORDER BY created_at DESC
            """
        )

        batches = cur.fetchall()

        # ====================================================
        # DEFECTS
        # ====================================================

        if batch_id:

            cur.execute(
                """
                SELECT *
                FROM defects
                WHERE batch_id = %s
                ORDER BY detected_at DESC
                """,
                (
                    batch_id,
                )
            )

        elif active_batch:

            cur.execute(
                """
                SELECT *
                FROM defects
                WHERE batch_id = %s
                ORDER BY detected_at DESC
                """,
                (
                    active_batch["batch_id"],
                )
            )

        else:

            cur.execute(
                """
                SELECT *
                FROM defects
                ORDER BY detected_at DESC
                LIMIT 100
                """
            )

        defects = cur.fetchall()

        # ====================================================
        # STATISTICS
        # ====================================================

        if batch_id:

            cur.execute(
                """
                SELECT
                    defect_name,
                    COUNT(*) AS count
                FROM defects
                WHERE batch_id = %s
                GROUP BY defect_name
                """,
                (
                    batch_id,
                )
            )

        elif active_batch:

            cur.execute(
                """
                SELECT
                    defect_name,
                    COUNT(*) AS count
                FROM defects
                WHERE batch_id = %s
                GROUP BY defect_name
                """,
                (
                    active_batch["batch_id"],
                )
            )

        else:

            cur.execute(
                """
                SELECT
                    defect_name,
                    COUNT(*) AS count
                FROM defects
                GROUP BY defect_name
                """
            )

        stats = cur.fetchall()

        labels = [
            row["defect_name"]
            .replace("_", " ")
            .title()
            for row in stats
        ]

        values = [
            row["count"]
            for row in stats
        ]

        cur.close()
        conn.close()

    except Exception as e:

        print(
            "Error:",
            e
        )

        defects = []
        batches = []
        labels = []
        values = []

    selected_batch = None

    if batch_id:

        try:
            selected_batch = int(
                batch_id
            )
        except:
            selected_batch = None

    return render_template(
        "defects.html",
        defects=defects,
        batches=batches,
        selected_batch=selected_batch,
        active_batch=active_batch,
        defect_stats={
            "labels": labels,
            "values": values
        }
    )


# ============================================================
# DELETE DEFECT
# ============================================================

@app.route(
    "/delete_defect/<int:defect_id>"
)
def delete_defect(defect_id):

    try:

        conn = get_db_connection()

        cur = conn.cursor()

        cur.execute(
            """
            SELECT image_path
            FROM defects
            WHERE id = %s
            """,
            (
                defect_id,
            )
        )

        row = cur.fetchone()

        if row:

            path = row[0].lstrip("/")

            if os.path.exists(path):

                os.remove(path)

            cur.execute(
                """
                DELETE FROM defects
                WHERE id = %s
                """,
                (
                    defect_id,
                )
            )

            conn.commit()

        cur.close()
        conn.close()

    except Exception as e:

        print(
            "Delete defect error:",
            e
        )

    redirect_url = url_for(
        "defects_list"
    )

    if request.args.get(
        "batch_id"
    ):

        redirect_url += (
            "?batch_id="
            + request.args.get(
                "batch_id"
            )
        )

    return redirect(
        redirect_url
    )


# ============================================================
# DOWNLOAD PDF
# ============================================================

@app.route("/download_pdf")
def download_pdf():

    batch_id = request.args.get(
        "batch_id"
    )

    try:

        conn = get_db_connection()

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        # ====================================================
        # GET DEFECT DATA
        # ====================================================

        if batch_id:

            cur.execute(
                """
                SELECT
                    d.*,
                    b.batch_name,
                    b.fabric_type
                FROM defects d
                JOIN batch_details b
                    ON d.batch_id = b.batch_id
                WHERE d.batch_id = %s
                ORDER BY d.detected_at DESC
                """,
                (
                    batch_id,
                )
            )

        else:

            cur.execute(
                """
                SELECT
                    d.*,
                    COALESCE(
                        b.batch_name,
                        'No Batch'
                    ) AS batch_name,
                    COALESCE(
                        b.fabric_type,
                        '-'
                    ) AS fabric_type
                FROM defects d
                LEFT JOIN batch_details b
                    ON d.batch_id = b.batch_id
                ORDER BY d.detected_at DESC
                """
            )

        defects = cur.fetchall()

        cur.close()
        conn.close()

        # ====================================================
        # CREATE PDF
        # ====================================================

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.8 * inch
        )

        elements = []

        styles = getSampleStyleSheet()

        title = (
            f"Fabric Defect Report - Batch #{batch_id}"
            if batch_id
            else "All Defects Report"
        )

        elements.append(
            Paragraph(
                title,
                styles["Title"]
            )
        )

        elements.append(
            Paragraph(
                "Generated: "
                + datetime.now().strftime(
                    "%d %B %Y %H:%M"
                ),
                styles["Normal"]
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                f"<b>Total Defects: {len(defects)}</b>",
                styles["Normal"]
            )
        )

        elements.append(
            Spacer(1, 20)
        )

        # ====================================================
        # TABLE
        # ====================================================

        if defects:

            data = [
                [
                    "#",
                    "Image",
                    "Type",
                    "Conf.",
                    "Batch",
                    "Time"
                ]
            ]

            for i, defect in enumerate(
                defects,
                1
            ):

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                try:

                    img_path = os.path.join(
                        os.getcwd(),
                        defect["image_path"].lstrip("/")
                    )

                    img = RLImage(
                        img_path,
                        width=1.6 * inch,
                        height=1.3 * inch
                    )

                    img = KeepInFrame(
                        1.6 * inch,
                        1.3 * inch,
                        [img]
                    )

                except Exception:

                    img = Paragraph(
                        "[No Image]",
                        styles["Normal"]
                    )

                # --------------------------------------------
                # ROW
                # --------------------------------------------

                data.append(
                    [
                        str(i),

                        img,

                        defect[
                            "defect_name"
                        ].replace(
                            "_",
                            " "
                        ).title(),

                        f"{defect['confidence'] * 100:.1f}%",

                        (
                            f"{defect['batch_name']} "
                            f"({defect['fabric_type']})"
                        ),

                        defect[
                            "detected_at"
                        ].strftime(
                            "%d %b %H:%M"
                        )
                    ]
                )

            table = Table(
                data,
                colWidths=[
                    0.4 * inch,
                    1.8 * inch,
                    1.6 * inch,
                    0.8 * inch,
                    2 * inch,
                    1.4 * inch
                ]
            )

            table.setStyle(
                TableStyle(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.HexColor(
                                "#0066cc"
                            )
                        ),

                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.white
                        ),

                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            1,
                            colors.grey
                        ),

                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER"
                        ),

                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "MIDDLE"
                        )
                    ]
                )
            )

            elements.append(
                table
            )

        else:

            elements.append(
                Paragraph(
                    "No defects found.",
                    styles["Normal"]
                )
            )

        # ====================================================
        # BUILD PDF
        # ====================================================

        doc.build(
            elements
        )

        buffer.seek(0)

        response = make_response(
            buffer.getvalue()
        )

        response.headers[
            "Content-Type"
        ] = "application/pdf"

        response.headers[
            "Content-Disposition"
        ] = (
            "attachment; "
            "filename=Defect_Report_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M"
            )
            + ".pdf"
        )

        return response

    except Exception as e:

        print(
            "PDF Error:",
            e
        )

        return (
            "Error generating PDF",
            500
        )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Starting WeaveCheck application...")
    print("Open: http://127.0.0.1:5001")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5001,
        threaded=True,
        debug=True
    )