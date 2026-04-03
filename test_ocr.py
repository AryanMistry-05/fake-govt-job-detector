import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract\tesseract.exe"

img = Image.open("temp.png")

print(pytesseract.image_to_string(img))