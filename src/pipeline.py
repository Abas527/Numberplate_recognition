import cv2
import re
from matplotlib import text
from ultralytics import YOLO
import easyocr
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Load trained YOLO model
model = YOLO(r"runs\detect\runs\numberplate_detector\weights\best.pt")

# Initialize EasyOCR for Nepali (Devanagari)
reader = easyocr.Reader(['ne'], gpu=True) 


def clean_text(text):
    # Keep only Devanagari characters + Nepali digits
    cleaned = re.sub(r'[^०-९]', '', text)
    return cleaned.strip()


def detect_and_read(image):
    img = image.copy()

    results = model(img)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Crop plate
            plate = img[y1:y2, x1:x2]

            # Resize
            plate = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray=cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            # increase contrast
            #gray=cv2.equalizeHist(gray)

            # filter
            #gray=cv2.bilateralFilter(gray, 11, 17, 17)

            # thresholding
            #gray=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            

            # Run EasyOCR
            ocr_result = reader.readtext(gray)

            detected_text = ""
            for bbox, text, confidence in ocr_result:
                detected_text += text

            cleaned_text = clean_text(detected_text)

            print("Detected Plate:", cleaned_text)

            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(r"fonts\NotoSansDevanagari.ttf",32)
            draw.text((x1+50,y1-50), cleaned_text, font=font, fill=(0,255,0))
            img = np.array(pil_img)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img, cleaned_text

    


if __name__ == "__main__":
    detect_and_read("test3.png")