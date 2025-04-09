import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import cv2
import pytesseract
import regex as re
import json

# Function to open a file dialog and select an image
def load_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title='Select Image', filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

# Define a function to crop the image for eAadhar
def crop_image(image_path):
    img = Image.open(image_path)

    # Get image dimensions (width, height)
    width, height = img.size

    # Define the coordinates for the area to crop (increasing the left part)
    left = int(0.02 * width)  # Increase the left part to capture more of the face
    top = int(0.70 * height)  # Keep the same height adjustment to remove more from the top
    right = int(0.45 * width) # Stop before reaching the middle
    bottom = int(1.00 * height) # Cover the full bottom part

    # Crop the image using the defined coordinates
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

class AadharInfoExtractor:
    def __init__(self):
        self.extracted = {}

    def find_aadhaar_number(self, ocr_text):
        aadhaar_number_patn = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        match = re.search(aadhaar_number_patn, ocr_text)
        if match:
            aadhaar_number = re.sub(r'\s+', ' ', match.group()).strip()
            return aadhaar_number
        return 'NAN'

    def find_name(self, ocr_text):
        """Function to find the name on the Aadhaar card when it's not explicitly mentioned."""
        lines = ocr_text.split('\n')
        possible_names = []

        for line in lines:
            clean_line = re.sub(r'[^A-Za-z\s]', '', line.strip())  # Clean the line

            # Skip lines with keywords related to other fields
            if any(keyword in clean_line.lower() for keyword in
                   ['government', 'india', 'dob', 'date of birth', 'male', 'female', 'address', 'issue','mobile no']):
                continue

            # Look for potential name (usually multiple words with capital letters)
            name_patn = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})'  # Capture names with 1 to 3 words
            matches = re.findall(name_patn, clean_line)

            for match in matches:
                if len(match.split()) <= 3:  # Limit to max 3 words
                    possible_names.append(match.strip())

        # Return the longest valid name found, or 'NAN' if none found
        if possible_names:
            return max(possible_names, key=len)

        return 'NAN'

    def find_dob(self, ocr_text):
        date_patn = r'\b\d{2}/\d{2}/\d{4}\b'
        for line in ocr_text.split('\n'):
            if 'dob' in line.lower() or 'date of birth' in line.lower():
                match = re.search(date_patn, line)
                if match:
                    return match.group()

        dates = re.findall(date_patn, ocr_text)
        return dates[0] if dates else 'NAN'

    def find_gender(self, ocr_text):
        ocr_text_lower = ocr_text.lower()
        if 'female' in ocr_text_lower:
            return 'Female'
        elif 'male' in ocr_text_lower:
            return 'Male'
        return 'NAN'

    def info_extractor(self, front_image):
        img = cv2.imread(front_image, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

        ocr_text = pytesseract.image_to_string(img, config='--psm 6')
        print(ocr_text)  # For debugging purposes

        aadhaar_no = self.find_aadhaar_number(ocr_text)
        name = self.find_name(ocr_text)
        dob = self.find_dob(ocr_text)
        gender = self.find_gender(ocr_text)

        self.extracted = {
            'Aadhaar_number': aadhaar_no,
            'Name': name,
            'Gender': gender,
            'DOB': dob,
        }

        return json.dumps(self.extracted)

if __name__ == '__main__':
    model = tf.keras.models.load_model("pvc_eaadhar_simplified.h5", compile=False)
    image_path = load_image()

    if image_path:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make a prediction
        prediction = model.predict(img_array)
        if prediction > 0.5:
            #print("Prediction: PVC Aadhar")
            # Directly extract information from PVC Aadhar
            extractor = AadharInfoExtractor()
            extracted_info = extractor.info_extractor(image_path)
            print(extracted_info)
        else:
            #print("Prediction: eAadhar")
            # Crop the image for eAadhar and then extract information
            cropped_image = crop_image(image_path)
            #cropped_image.show()  # Display the cropp  ed image for testing

            # Save the cropped image temporarily for extraction
            cropped_image_path = "temp_eAadhar.jpg"
            cropped_image.save(cropped_image_path)

            extractor = AadharInfoExtractor()
            extracted_info = extractor.info_extractor(cropped_image_path)
            print(extracted_info)
    else:
        print("No image selected.")
