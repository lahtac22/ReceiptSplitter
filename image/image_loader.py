import cv2
import fitz
import numpy as np

from config import PDF_DPI


def load_image(file):
    """
    Load either an image or the first page of a PDF.

    Returns:
        image (numpy array)
    """

    if file.suffix.lower() == ".pdf":
        return load_pdf(file)

    image = cv2.imread(str(file))

    if image is None:
        raise RuntimeError(f"Unable to read image: {file}")

    return image


def load_pdf(file):
    """
    Convert the first page of a PDF to an OpenCV image.
    """

    doc = fitz.open(file)

    page = doc.load_page(0)

    pix = page.get_pixmap(dpi=PDF_DPI)

    img = np.frombuffer(pix.samples, dtype=np.uint8)

    image = img.reshape(pix.height, pix.width, pix.n)

    # Convert RGB → BGR for OpenCV
    if pix.n == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    elif pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    doc.close()

    return image