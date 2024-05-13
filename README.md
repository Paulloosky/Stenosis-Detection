### README File

---

### Project Title:

###### Integration of Machine Learning to Improve MRI-Based Cervical Spine Disease Diagnosis

---

### Aims and Objectives

The aim of this project is to develop deep learning models for the detection of stenosis, as well as the classification
of its severity. The main thrust of this project is to:

1. Do this with pure automated annotation (i.e., minimal to no expert annotation).
2. To deploy the trained model as a simple prototype.

---

### Project Overview

The project was carried out with a total of 40 patient MRI volumes. Two of them (patients 17 and 35) were noisy or missing.

1. __Image Extraction__: Images are extracted from the MRI volumes via the _Slice Sweep_ utility in _3D Slicer_.
2. __Image Annotation__: Extracted images are annotated in _Label Studio_.
3. __Image Segmentation__: YOLO v8 and v9 segmentation models are trained on annotated images.
4. __Segmentation Mask Prediction__: Segmentation masks are predicted for each image using the trained segmentation models.
5. __Stenosis Bounding Box and Grade Generation__: The masks are leveraged for generating bounding boxes for stenosis detection, as well as the apppropriate stenosis grade.
6. __Stenosis Detection__: YOLO v8 and v9 detection models are trained on the images using the generated labels.
7. __Model Deployment__: The trained models are deployed on local machine via __Streamlit__.

---

### Requirements

Software requirements for this project include:

1. 3D Slicer (for image extraction)
2. Pydicom (for medical image operations)
3. Streamlit (for model deployment)
4. OpenCV (for image operations)
5. Sci-kit Learn (for evaluation metrics)

Other requirements can be found in the [requirements.txt](requirements.txt) file.


---

### Deployment

Deplying the model is as simple as running the following command from the terminal:

```bash
streamlit run deployment.py

```
Samples of the model in use are shown in the screenshots below:



![Deployment Landing Page](Deployment_Landing_Page.png)



![Folder Upload](Drag_and_Drop.png)




![Displayed Model Predictions](Displayed_Predictions.png)

---

### Repository Structure

The repository structure for this project is shown below:

```angular2html
.
├── MRI.jpg
|
├── README.md
|
├── config       # YOLO configuration files
│   ├── segmentation_config.yaml
│   └── stenosis_config.yaml
|
├── datasets     # Segementation and stenosis detection datasets
│   ├── segmentation_dataset
│   └── stenosis_dataset
|
├── deployment.py  # Deployment script
|
├── extras
│   ├── classes.txt
│   ├── coco_annotations.json
│   ├── extra_coco_annotations.json
│   └── notes.json
|
├── stenosis_annotation  # Notebook for stenosis label generation
│   └── final_stenosis_annotation.ipynb
|
├── yolov8_detection
│   ├── runs
│   ├── c2085311_yolo_v8_default_train.ipynb          # YOLOv8 in default mode
│   ├── c2085311_yolo_v8_dropout_train.ipynb          # YOLOv8 in dropout mode
│   └── c2085311_yolo_v8_freeze_train.ipynb           # YOLOv8 in weight freezing mode
|
├── yolov8_segmentation
│   ├── runs
│   ├── c2085311_yolo_v8_default_train.ipynb          # YOLOv8 in default mode
│   ├── c2085311_yolo_v8_dropout_train.ipynb          # YOLOv8 in dropout mode
│   ├── c2085311_yolo_v8_dropout_evaluation.ipynb     # Evaluation notebook for YOLOv8 in dropout mode
│   ├── c2085311_yolo_v8_default_evaluation.ipynb     # Evaluation notebook for YOLOv8 in default mode
│   ├── c2085311_yolo_v8_freeze_train.ipynb           # YOLOv8 in weight freezing mode
│   └── c2085311_yolo_v8_freeze_evaluation.ipynb      # Evaluation notebook for YOLOv8 in weight freezing mode
|
├── yolov9_detection
│   ├── runs
│   ├── c2085311_yolo_v9_default_train.ipynb          # YOLOv9 in default mode
│   ├── c2085311_yolo_v9_dropout_train.ipynb          # YOLOv9 in dropout mode
│   └── c2085311_yolo_v9_freeze_train.ipynb           # YOLOv9 in weight freezing mode
|
└── yolov9_segmentation
    ├── runs
    ├── c2085311_yolo_v9_default_train.ipynb           # YOLOv9 in default mode
    ├── c2085311_yolo_v9_dropout_train.ipynb           # YOLOv9 in dropout mode
    ├── c2085311_yolo_v9_dropout_evaluation.ipynb      # Evaluation notebook for YOLOv9 in dropout mode
    ├── c2085311_yolo_v9_default_evaluation.ipynb      # Evaluation notebook for YOLOv9 in default mode
    ├── c2085311_yolo_v9_freeze_train.ipynb            # YOLOv9 in weight freezing mode
    └── c2085311_yolo_v9_freeze_evaluation.ipynb       # Evaluation notebook for YOLOv9 in weight freezing mode
```

---

### Directory Key
The purposes for all files and subdirectories are as shown below:

<table>
    <tr align="center">
        <td><strong>Identifier</strong></td>
        <td><strong>Subdirectory?</strong></td>
        <td><strong>Purpose</strong></td>
    </tr>
    <tr align="center">
        <td>MRI.jpg</td>
        <td>No</td>
        <td>Banner Image for Streamlit Deployment</td>
    </tr>
    <tr align="center">
        <td>README.md</td>
        <td>No</td>
        <td>Information on entire project repository</td>
    </tr>
    <tr align="center">
        <td>config</td>
        <td>Yes</td>
        <td>Contains configuration .YAML files for YOLO segmentation and stenosis detection tasks</td>
    </tr>
    <tr align="center">
        <td>datasets</td>
        <td>Yes</td>
        <td>Contains two subdirectories: one for the segmentation dataset, another for the stenosis detection dataset</td>
    </tr>
    <tr align="center">
        <td>deployment.py</td>
        <td>No</td>
        <td>Python script for model deployment on Streamlit</td>
    </tr>
    <tr  align="center">
        <td>extras</td>
        <td>Yes</td>
        <td>Contains extra .JSON and .TXT files for annotations. These files are exported from Label Studio.</td>
    </tr>
    <tr align="center">
        <td>stenosis_annotation</td>
        <td>Yes</td>
        <td>Contains the notebook used for generating (a). bounding boxes for stenosis detection and (b). severity labels.</td>
    </tr>
    <tr align="center">
        <td>yolov8_detection</td>
        <td>Yes</td>
        <td>Notebooks for YOLO v8 + stenosis detection (default, dropout, and freeze modes)</td>
    </tr>
    <tr align="center">
        <td>yolov8_segmentation</td>
        <td>Yes</td>
        <td>Notebooks for training YOLO v8 instance segmentation models (default, dropout, and freeze modes)</td>
    </tr>
    <tr align="center">
        <td>yolov9_detection</td>
        <td>Yes</td>
        <td>Notebooks for YOLO v9 + stenosis detection (default, dropout, and freeze modes)</td>
    </tr>
    <tr align="center">
        <td>yolov9_segmentation</td>
        <td>Yes</td>
        <td>Notebooks for training YOLO v9 instance segmentation models (default, dropout, and freeze modes)</td>
    </tr>
    <tr align="center">
        <td>yolov[number]_[task]/runs</td>
        <td>Yes</td>
        <td>Contains the results from training and evaluation of the YOLO models for the detection or segmentation 
tasks, as the case may be.</td>
    </tr>
</table>
