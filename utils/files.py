from config import INPUT_DIR


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf"
}


def get_input_files():

    files = []

    for file in INPUT_DIR.iterdir():

        if file.suffix.lower() in IMAGE_EXTENSIONS:

            files.append(file)

    return sorted(files)