
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import easyocr
import pandas as pd
from datetime import datetime

# -----------------------------
# LOAD MODELS
# -----------------------------

vehicle_model = YOLO("model1_best.pt")
violation_model = YOLO("model2_best.pt")

reader = easyocr.Reader(['en'])

MODEL1_CLASSES = {
    0: "Person",
    1: "Motorcycle",
    2: "Car",
    3: "Bus",
    4: "Truck"
}

MODEL2_CLASSES = {
    0: "Helmet",
    1: "No Helmet",
    2: "Seatbelt",
    3: "Plate"
}

# -----------------------------
# UI
# -----------------------------

st.title("🚦 Traffic Violation Detection System")

uploaded_file = st.file_uploader(
    "Upload Traffic Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PROCESS IMAGE
# -----------------------------

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    )

    temp_file.write(uploaded_file.read())
    temp_file.close()

    image_path = temp_file.name

    image = cv2.imread(image_path)

    st.write("Processing image...")

    vehicle_results = vehicle_model(image_path)
    violation_results = violation_model(image_path)

    st.success("Detection completed")

    # -----------------------------
    # VEHICLE DETECTIONS
    # -----------------------------

    st.subheader("🚗 Vehicle Detections")

    vehicle_classes = []

    for box in vehicle_results[0].boxes:

        cls = int(box.cls[0])

        vehicle_classes.append(cls)

        st.write(
            MODEL1_CLASSES.get(
                cls,
                f"Unknown ({cls})"
            )
        )

    # -----------------------------
    # VIOLATION OBJECTS
    # -----------------------------

    st.subheader("🛑 Violation Object Detections")

    violation_classes = []

    for box in violation_results[0].boxes:

        cls = int(box.cls[0])

        violation_classes.append(cls)

        st.write(
            MODEL2_CLASSES.get(
                cls,
                f"Unknown ({cls})"
            )
        )

    # -----------------------------
    # RULE ENGINE
    # -----------------------------

    motorcycle_detected = 1 in vehicle_classes
    car_detected = 2 in vehicle_classes

    helmet_detected = 0 in violation_classes
    no_helmet_detected = 1 in violation_classes

    seatbelt_detected = 2 in violation_classes

    violations = []

    if motorcycle_detected and no_helmet_detected:
        violations.append("Helmet Violation")

    if car_detected and not seatbelt_detected:
        violations.append("Possible Seatbelt Violation")

    if len(violations) == 0:
        violations.append("No Violation")

    st.subheader("⚠ Violation Report")

    for v in violations:
        if v == "No Violation":
            st.success(v)
        else:
            st.error(v)

    # -----------------------------
    # OCR
    # -----------------------------

    plate_number = "Not Detected"

    for box in violation_results[0].boxes:

        cls = int(box.cls[0])

        if cls == 3:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            plate_crop = image[y1:y2, x1:x2]

            try:

                result = reader.readtext(
                    plate_crop
                )

                if len(result) > 0:
                    plate_number = result[0][1]

            except:
                pass

            break

    st.subheader("🔤 License Plate")

    st.info(plate_number)

    # -----------------------------
    # EVIDENCE REPORT
    # -----------------------------

    report = {
        "Timestamp":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "Plate Number":
        plate_number,

        "Violation":
        ", ".join(violations)
    }

    st.subheader("📄 Evidence Report")

    st.json(report)

    # -----------------------------
    # DOWNLOAD CSV
    # -----------------------------

    df = pd.DataFrame([report])

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="violation_report.csv",
        mime="text/csv"
    )

    # -----------------------------
    # ANNOTATED IMAGE
    # -----------------------------

    st.subheader("📷 Annotated Output")

    annotated = violation_results[0].plot()

    st.image(
        annotated,
        caption="Detected Objects",
        use_container_width=True
    )

