# Import libraries

import numpy as np
import cv2
import pydicom

from PIL import Image

import streamlit as st
from ultralytics import YOLO

import warnings

# Ignore all non-failing warnings
warnings.filterwarnings("ignore")

# Set visualization colors
COLORS = [
    (0, 255, 250),  # Grade 0 Stenosis
    (255, 200, 0),  # Grade 1 Stenosis
    (0, 40, 255),   # Grade 2 Stenosis
    (200, 40, 255)  # Grade 3 Stenosis
]

# Page title, page icon, page loading state
st.set_page_config(
    page_title="Cervical Spinal Stenosis Detection",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

# Hide unnecessary parts of the code
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(
    hide_streamlit_style,
    unsafe_allow_html=True
)

# Create a sidebar on the left
# This sidebar will contain some page styling and information
with st.sidebar:
    st.image('MRI.jpg')
    st.title("Cervical Stenosis Detection")
    st.subheader(
        "Accurate detection of stenosis present in the sagittal view of spinal MRI scans. "
        "This helps a user (usually a medical practitioner) to easily detect and provide accurate prognosis."
    )

# Set the main title of the deployment page
st.write("""# Cervical Stenosis Detection""")

# Load the trained model and cache it in memory
@st.cache_resource
def load_model(model_choice = "yolov8", sub_choice = "default"):
    """Load trained stenosis detection model"""
    assert model_choice in ["yolov8", "yolov9"], "`model_choice` must be one off `yolov8` and `yolov9`."

    assert sub_choice in ["default", "dropout", "freeze"], "`sub_choice` must be one of `default`, `dropout` or `freeze`."

    suffix = "s" if model_choice == "yolov8" else "c"
    sub_choice = "" if sub_choice == "default" else "_" + sub_choice

    return YOLO(f'{model_choice}_detection/runs/detect/stenosis_{model_choice}{suffix}{sub_choice}/weights/best.pt')


# Select the detection model of choice
model_choice = st.selectbox(
    label = "Select detection model of choice: YOLO v8 or YOLO v9: ",
    options = ["YOLO v8", "YOLO v9"]
)

model_sub_choice = st.selectbox(
    label = "Select detection model subtype of choice: dropout, freeze, or default: ",
    options = ["Default", "Dropout", "Freeze"]
)

model_choice = model_choice.lower().replace(" ", "")
model_sub_choice = model_sub_choice.lower().replace(" ", "")

# Run the code to load the model into memory
with st.spinner('Model is being loaded..'):
    model=load_model(
        model_choice = model_choice,
        sub_choice = model_sub_choice
    )

# def extract_spine_images(folder)


# Import image files
files = st.file_uploader(
    "Drag and drop the T2-weighted spine MRI folder for any patient here:",
    accept_multiple_files = True, type=["jpg", "png", "ima"]
)

# Filter out the images between timesteps 7 and 10 (they seem to be clearer)
files = (
    [f for f in files if int(f.name.split(".")[0].split("_")[-1]) in [8]]
    if any(filter(lambda x: x.name.endswith(".ima"), files)) else files
)
files = [f for f in files if ("T2" in f.name) and ("SAG" in f.name)]

# Sort the loaded images according to timestep
files = sorted(files, key = lambda x: int(x.name.split('.')[0].split('_')[-1].lstrip('0')))


# Print information as to color coding for stenosis grades
st.info(
    body = "Color Code:\n\nGrade 0 Stenosis = Light Blue"
           "\n\nGrade 1 Stenosis = Yellow\n\nGrade 2 Stenosis = Blue"
           "\n\nGrade 3 Stenosis = Light Purple",
    icon = "🤖"
)


def image_preprocessing(uploaded_file):
    """Preprocess uploaded StreamLit file."""
    if uploaded_file.name.endswith(".ima"):
        # Read DICOM image
        dicom_image = pydicom.dcmread(uploaded_file).pixel_array

        # Normalize DICOM image
        dicom_image = ((dicom_image - np.min(dicom_image)) / (np.max(dicom_image) - np.min(dicom_image))) * 255
        # Resize image
        loaded_image = cv2.resize(dicom_image.astype(np.uint8), (694, 542))
    else:
        loaded_image = np.array(Image.open(uploaded_file))

    return loaded_image


def generate_predictions(image, model):
    """Pass an image through the trained model and return the predicted results."""
    img = np.asarray(image) if not isinstance(image, np.ndarray) else image

    if len(img.shape) < 3:
        img = np.stack([img, img, img], axis = -1)
    prediction = model.predict(img)
    return prediction[0]


def postprocessingv1(prediction):
    """Take the predicted output of the model and format it for further use."""
    boxes = prediction.boxes
    return (prediction.plot(), False) if boxes.shape[0] != 0 else (prediction.orig_img, True)


def extract_bbox(result):
    """Extract coordinates for displaying bounding boxes."""
    info = result.summary()

    class_names = [r['name'] for r in info]
    class_indices = [r['class'] for r in info]

    bbox_info = [r['box'] for r in info]
    bbox_info = [
        [(int(b['x1']), int(b['y1'])), (int(b['x2']), int(b['y2']))]
        for b in bbox_info
    ]
    return [
        {
            "name": name,
            "index": index,
            "box": box
        }
        for name, index, box in zip(class_names, class_indices, bbox_info)
    ]


def generate_image_with_bounding_box(result):
    extracted_result = extract_bbox(result)
    orig_img = result.orig_img
    class_names = []

    for bbox in extracted_result:
        upper_left, lower_right = bbox['box']
        class_name = bbox['name']
        class_names.append(class_name)
        class_index = bbox['index']

        color = COLORS[class_index]

        lower_left = (int(upper_left[0]), int(lower_right[1]))
        upper_right = (int(lower_right[0]), int(upper_left[1]))

        orig_img = cv2.line(orig_img, upper_left, lower_left, color=color)
        orig_img = cv2.line(orig_img, lower_left, lower_right, color=color)
        orig_img = cv2.line(orig_img, lower_right, upper_right, color=color)
        orig_img = cv2.line(orig_img, upper_right, upper_left, color=color)

    return orig_img, class_names


def postprocessingv2(prediction):
    """Take the predicted output of the model and format it for further use."""
    boxes = prediction.boxes
    return generate_image_with_bounding_box(prediction) if boxes.shape[0] != 0 else (prediction.orig_img, None)


# Generate predictions for all image files loaded
for i, file in enumerate(files):
    # print("File properties:", dir(file))
    if file is None:
        st.text("Please upload an image file: ")
    else:
        if i == 0:
            ID = file.name.split(".")[0].split("_")[-2].lstrip("0")
            st.info(f"Diagnosis for Patient {ID}:")

        timestep = int(file.name.split('.')[0].split('_')[-1].lstrip('0'))

        # Load image
        image_ = image_preprocessing(file)
        # print(np.array(image).astype(np.uint8).max())

        image = np.array(image_).astype(np.uint8)

        # Generate predictions from model
        predictions = generate_predictions(image, model)

        # Postprocess predictions
        results, detections = postprocessingv2(predictions)

        st.image(
            results,
            caption=f"MRI Slice for patient {ID} at timestep, t = {timestep}",
            use_column_width=True,
            clamp=True,
        )

        # Display results of prediction
        # x = random.randint(98, 99) + random.randint(0, 99) * 0.01
        # st.sidebar.error("Accuracy : " + str(x) + " %")

        prefix = f"Patient Diagnosis at timestep, t = {timestep}:\n\n"

        if detections is None:
            string = "No Stenosis Detected! Please check and confirm."
            st.balloons()
            st.sidebar.success(prefix + string)

        else:
            class_detections = list(set(detections))
            if len(class_detections) == 1:
                message = class_detections[0] + " Detected!"
            elif len(class_detections) == 2:
                message = " and ".join([t.replace(" Stenosis", "") for t in class_detections])
                message = message + " Stenosis Detected!"
            else:
                message = ", ".join([t.replace(" Stenosis", "") for t in class_detections[:-1]])
                message = message + ", and " + class_detections[-1] + " Detected!"

            string = f"{message}\n\nPlease check and confirm."
            st.sidebar.warning(prefix + string)
            st.markdown("## Remedy")
            st.info(
                "Please take appropriate medical action!"
            )
