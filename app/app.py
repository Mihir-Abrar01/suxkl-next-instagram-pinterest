
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    abort,
    send_from_directory
)
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
OBJECTS_DIR = BASE_DIR / "objects"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OBJECTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["UPLOAD_PATH"] = str(UPLOAD_DIR)
app.config["OBJECTS_PATH"] = str(OBJECTS_DIR)
app.config["UPLOAD_EXTENSIONS"] = {".jpg", ".jpeg", ".png", ".gif"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

app.config["FILE_PATH"] = None
app.config["FILE_NAME"] = []
app.config["CONFIDENT_RANG"] = 0.5

# Stores detected object images during the current process.
FILE_OBJ = {}


# --------------------------------------------------
# COCO OBJECT CLASSES
# --------------------------------------------------

OBJECTS = [
    "__background__", "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


OBJECTS_html = [
    "all", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# --------------------------------------------------
# LOAD FASTER R-CNN MODEL
# --------------------------------------------------

print("Loading Faster R-CNN model...")

model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True
)

model_.eval()

for param in model_.parameters():
    param.requires_grad = False

print("Faster R-CNN model loaded successfully.")

transform = transforms.Compose([
    transforms.ToTensor()
])


def model(image_tensor):
    with torch.no_grad():
        prediction = model_(image_tensor)

    return prediction


# --------------------------------------------------
# OBJECT DETECTION FUNCTIONS
# --------------------------------------------------

def get_predictions(pred, threshold=0.5, objects=None):

    predicted_classes = []

    labels = pred[0]["labels"].cpu().numpy()
    scores = pred[0]["scores"].cpu().numpy()
    boxes = pred[0]["boxes"].cpu().numpy()

    for label, probability, box in zip(labels, scores, boxes):

        if probability < threshold:
            continue

        object_name = OBJECTS[int(label)]

        if object_name == "N/A" or object_name == "__background__":
            continue

        if objects and object_name not in objects:
            continue

        predicted_classes.append((
            object_name,
            float(probability),
            [
                (float(box[0]), float(box[1])),
                (float(box[2]), float(box[3]))
            ]
        ))

    return predicted_classes


def draw_box(predicted_classes, image):

    image_array = np.array(image.convert("RGB"))

    image_array = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    for predicted_class in predicted_classes:

        object_name = predicted_class[0]
        probability = predicted_class[1]
        box = predicted_class[2]

        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[1][0])
        y2 = int(box[1][1])

        label = f"{object_name}: {probability:.2f}"

        cv2.rectangle(
            image_array,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        text_y = max(y1 - 10, 20)

        cv2.putText(
            image_array,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

    return image_array


# --------------------------------------------------
# FILE MANAGEMENT
# --------------------------------------------------

def get_uploaded_images():

    files = []

    for file in UPLOAD_DIR.iterdir():

        if (
            file.is_file()
            and file.suffix.lower() in app.config["UPLOAD_EXTENSIONS"]
        ):
            files.append(file.name)

    return sorted(
        files,
        key=lambda name: (UPLOAD_DIR / name).stat().st_mtime,
        reverse=True
    )


def get_latest_uploaded_image():

    files = get_uploaded_images()

    if not files:
        return None

    return UPLOAD_DIR / files[0]


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    # IMPORTANT:
    # Read actual files from disk instead of relying on
    # temporary in-memory FILE_NAME state.

    files = get_uploaded_images()

    object_files = [
        file.name
        for file in OBJECTS_DIR.iterdir()
        if (
            file.is_file()
            and file.suffix.lower() in app.config["UPLOAD_EXTENSIONS"]
        )
    ]

    return render_template(
        "index.html",
        files=files,
        objects_list=OBJECTS_html,
        object_files=object_files
    )


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

@app.route("/", methods=["POST"])
def upload_file():

    uploaded_file = request.files.get("file")

    if not uploaded_file or not uploaded_file.filename:
        return redirect(url_for("home"))

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        abort(400, description="Invalid filename.")

    file_extension = Path(filename).suffix.lower()

    if file_extension not in app.config["UPLOAD_EXTENSIONS"]:
        abort(400, description="Unsupported image format.")

    file_path = UPLOAD_DIR / filename

    # Avoid overwriting an existing image with the same name.
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{Path(filename).stem}_{timestamp}{file_extension}"
        file_path = UPLOAD_DIR / filename

    try:

        uploaded_file.save(str(file_path))

        # Verify that the uploaded file is a readable image.
        with Image.open(file_path) as uploaded_image:
            uploaded_image.verify()

    except Exception:

        if file_path.exists():
            file_path.unlink()

        abort(400, description="The uploaded file is not a valid image.")

    app.config["FILE_NAME"] = [filename]
    app.config["FILE_PATH"] = file_path

    print(f"Image uploaded successfully: {filename}")

    return redirect(url_for("home"))


# --------------------------------------------------
# FIND OBJECTS
# --------------------------------------------------

@app.route("/find_object", methods=["POST"])
def find():

    # Recover image from disk if application memory was reset.

    uploaded_images = get_uploaded_images()

    if not uploaded_images:
        return redirect(url_for("home"))

    current_path = app.config.get("FILE_PATH")

    if current_path:
        current_path = Path(current_path)
    else:
        current_path = None

    if current_path is None or not current_path.exists():

        latest_image = get_latest_uploaded_image()

        if latest_image is None:
            return redirect(url_for("home"))

        current_path = latest_image

        app.config["FILE_PATH"] = current_path
        app.config["FILE_NAME"] = [current_path.name]

    selected_object = request.form.get("objects", "all")

    if selected_object not in OBJECTS_html:
        abort(400, description="Invalid object selection.")

    confident_range = request.form.get("confident_range", "50")

    try:

        confidence = int(confident_range)

        if confidence < 1 or confidence > 100:
            raise ValueError

    except ValueError:

        abort(400, description="Invalid confidence threshold.")

    threshold = confidence / 100.0

    app.config["CONFIDENT_RANG"] = threshold

    print(f"Starting object detection: {current_path.name}")
    print(f"Selected object: {selected_object}")
    print(f"Confidence threshold: {threshold}")

    try:

        # Open the original uploaded image.
        image = Image.open(current_path).convert("RGB")

        # Convert image to PyTorch tensor.
        image_tensor = transform(image)

        # Run Faster R-CNN.
        prediction = model(
            torch.unsqueeze(image_tensor, 0)
        )

        # Filter predictions.
        if selected_object == "all":

            predicted_objects = get_predictions(
                prediction,
                threshold=threshold
            )

        else:

            predicted_objects = get_predictions(
                prediction,
                threshold=threshold,
                objects=[selected_object]
            )

        print(f"Objects detected: {len(predicted_objects)}")

        # Draw bounding boxes.
        image_numpy = draw_box(
            predicted_objects,
            image
        )

        # Generate result filename.
        filename = current_path.stem
        file_extension = current_path.suffix

        new_file_name = f"{filename}_object{file_extension}"

        new_file_path = OBJECTS_DIR / new_file_name

        # Save annotated image.
        success = cv2.imwrite(
            str(new_file_path),
            image_numpy
        )

        if not success:
            raise RuntimeError("Could not save detection result.")

        # Save individual detected object crops.
        for obj in predicted_objects:

            object_name = obj[0]

            x1 = int(obj[2][0][0])
            y1 = int(obj[2][0][1])
            x2 = int(obj[2][1][0])
            y2 = int(obj[2][1][1])

            # Keep coordinates within image boundaries.
            x1 = max(0, min(x1, image.width))
            x2 = max(0, min(x2, image.width))
            y1 = max(0, min(y1, image.height))
            y2 = max(0, min(y2, image.height))

            if x2 <= x1 or y2 <= y1:
                continue

            timestamp = datetime.now().strftime(
                "_%d_%m_%Y_%H_%M_%S_%f"
            )

            object_filename = (
                f"{object_name}{timestamp}{file_extension}"
            )

            object_file_path = OBJECTS_DIR / object_filename

            cropped_image = image.crop(
                (x1, y1, x2, y2)
            )

            cropped_image.save(object_file_path)

            if object_name not in FILE_OBJ:
                FILE_OBJ[object_name] = []

            FILE_OBJ[object_name].append(object_filename)

        # Release image memory.
        image.close()

        app.config["FILE_NAME"] = []
        app.config["FILE_PATH"] = None

        return render_template(
            "find_object.html",
            objects=selected_object,
            file=new_file_name,
            title=selected_object,
            range1=confidence
        )

    except Exception as error:

        app.logger.exception(
            "Object detection failed: %s",
            error
        )

        return (
            f"Object detection failed: {str(error)}",
            500
        )


# --------------------------------------------------
# OBJECT GALLERY
# --------------------------------------------------

@app.route("/your_object")
def your_gallery():

    return render_template(
        "your_object.html",
        obj_files=FILE_OBJ
    )


@app.route("/your_object/<galleryName>")
def view_object(galleryName):

    if galleryName not in FILE_OBJ:
        abort(404)

    return render_template(
        "view_obejct.html",
        obj_files=FILE_OBJ[galleryName],
        title=galleryName
    )


# --------------------------------------------------
# IMAGE SERVING ROUTES
# --------------------------------------------------

@app.route("/uploads/<filename>")
def upload(filename):

    return send_from_directory(
        app.config["UPLOAD_PATH"],
        filename
    )


@app.route("/objects/<filename>")
def upload_objects(filename):

    return send_from_directory(
        app.config["OBJECTS_PATH"],
        filename
    )


# --------------------------------------------------
# UPLOADED IMAGE GALLERY
# --------------------------------------------------

@app.route("/your_galary")
def view_gallery():

    files = get_uploaded_images()

    return render_template(
        "your_galary.html",
        obj_files=files
    )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False
    )
