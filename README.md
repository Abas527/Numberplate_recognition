
#  Nepali Number Plate Recognition (ANPR)

A deep learning–based **Automatic Number Plate Recognition (ANPR)** system built for **Nepali number plates (Devanagari digits only)** using:

*  YOLOv8 for plate detection
*  EasyOCR for Nepali digit recognition
*  Pillow for Unicode text rendering
*  Streamlit for interactive web UI

---

## 📌 Project Overview

This project detects Nepali vehicle number plates from images and extracts **Nepali digits (०–९)** using a custom-trained YOLO model combined with OCR.

The system pipeline:

```
Image Upload → YOLOv8 Detection → Plate Crop → Preprocessing → EasyOCR → Digit Cleaning → Display Results
```

---

##  Tech Stack

| Component             | Technology |
| --------------------- | ---------- |
| Object Detection      | YOLOv8     |
| OCR Engine            | EasyOCR    |
| Image Processing      | OpenCV     |
| Web App               | Streamlit  |
| Deep Learning Backend | PyTorch    |

---

## ✨ Features

✅ Detect Nepali number plates in images
✅ Extract only Nepali digits (०–९)
✅ Unicode rendering support
✅ Confidence-based OCR filtering
✅ Image preprocessing for improved accuracy
✅ Interactive web interface
✅ Modular pipeline design

---

## 📂 Project Structure

```
numberplate_recognition/
│
├── src/
│   └── pipeline.py
│
├── runs/
│   └── detect/
│       └── runs/numberplate_detector/
│                └── weights/
│                    └── best.pt
│
├── fonts/
│   └── NotoSansDevanagari.ttf
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Abas527/Numberplate_recognition.git
cd nepali-anpr
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

##  How It Works

### 1️⃣ Detection

* YOLOv8 detects bounding box around number plate.

### 2️⃣ Preprocessing

* Resize (2.5x scaling)
* Grayscale conversion
* Histogram equalization
* Adaptive thresholding

### 3️⃣ OCR

* EasyOCR extracts text
* Regex filters only Nepali digits
* Confidence threshold applied

---

##  Nepali Digit Filtering

Regex used:

```python
r'[^०-९]'
```

Only digits ०–९ are retained.

---

## 📈 Accuracy Improvements Implemented

* Crop padding
* Contrast enhancement
* Adaptive thresholding
* Confidence filtering
* Enlarged plate scaling

Expected prototype accuracy: **80–90%** on clear images.

---

## 🚀 Future Improvements

*  Real-time webcam detection
*  SQLite database logging
*  OCR accuracy metrics
*  Docker containerization
*  Cloud deployment
*  Custom OCR fine-tuning

---

## Example Output

Input → Vehicle image
Output → Detected plate with Nepali digits overlay

---

## 🎯 Use Cases

* Parking management systems
* Traffic monitoring
* Smart toll booths
* Security checkpoints
* Vehicle tracking systems

---

## 🛠️ Requirements

```
numpy==1.26.4
opencv-python==4.8.1.78
torch
ultralytics
easyocr
streamlit
pillow
```

---

## 👨‍💻 Author

Developed as a deep learning portfolio project demonstrating:

* Object Detection
* OCR Integration
* Image Processing
* End-to-End ML Deployment
* Web App Development

---

## 📜 License

MIT License

---

# ⭐ If You Like This Project

Give it a star on GitHub and feel free to fork or contribute!

---

