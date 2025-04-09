# Aadhaar Card Information Extractor

🎯 Project Goal:  
This project uses a pre-trained CNN model to classify Aadhaar card images as either **PVC** or **eAadhar**, and extracts essential details such as Aadhaar number, Name, Date of Birth, and Gender using OCR (Tesseract).

---

🧠 Key Features:
- Classifies Aadhaar image type (PVC or eAadhar).
- Crops and preprocesses eAadhar images for better OCR accuracy.
- Uses Tesseract OCR to extract:
  - Aadhaar Number
  - Name
  - Gender
  - Date of Birth
- Interactive file selection using a GUI.
- Designed with real-world data handling in mind.

---

🛠 Technologies Used:
- TensorFlow / Keras – for CNN model
- OpenCV & PIL – for image processing
- pytesseract – for Optical Character Recognition (OCR)
- regex – for accurate field extraction
- Tkinter – for GUI-based image selection

---

🚀 How to Run:

1. Install dependencies  
   Make sure you have Python 3.7+ and install the required packages using:

   pip install -r requirements.txt

2. Install Tesseract OCR Engine
   - Download from: https://github.com/tesseract-ocr/tesseract
   - Add it to your system PATH or configure the path inside your script if needed.

3. Run the script:

   python your_script_name.py

4. Select an Aadhaar image when prompted.

5. The script will:
   - Predict image type using a CNN model (pvc_eaadhar_simplified.h5)
   - If eAadhar, crop the face region for better OCR
   - Extract and display Aadhaar details in JSON format

---

📁 Files:
- Aadhar_Information_Extraction.py – Main script
- pvc_eaadhar_simplified.h5 – Pre-trained Keras model for Aadhaar type classification
- requirements.txt – Required Python libraries

---

📌 Example Output:

{
  "Aadhaar_number": "1234 5678 9012",
  "Name": "Ravi Kumar",
  "Gender": "Male",
  "DOB": "12/03/1998"
}

---

🙋‍♂️ Author:
Dinesh
Data Science Trainee | Python & ML Enthusiast  
GitHub: https://github.com/dinesh4166
