from pathlib import Path


def write_report(
    folder,
    file_name,
    image,
    rectangles,
    gaps
):
    """
    Write an inspection report for one processed file.
    """

    report_file = folder / "report.txt"

    width = image.shape[1]
    height = image.shape[0]

    with open(report_file, "w", encoding="utf-8") as report:

        report.write("Receipt Inspector Report\n")
        report.write("========================\n\n")

        report.write(f"File: {file_name}\n\n")

        report.write("Image\n")
        report.write("------------------------\n")
        report.write(f"Width  : {width} px\n")
        report.write(f"Height : {height} px\n\n")

        report.write("Detection\n")
        report.write("------------------------\n")
        report.write(f"Receipt Candidates : {len(rectangles)}\n")
        report.write(f"Vertical Gaps      : {len(gaps)}\n")