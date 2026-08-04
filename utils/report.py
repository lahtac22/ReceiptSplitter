from datetime import datetime


def write_heading(report, title):

    report.write(title + "\n")
    report.write("=" * len(title) + "\n\n")


def write_section(report, title):

    report.write(title + "\n")
    report.write("-" * len(title) + "\n")


def write_report(
    folder,
    file_name,
    image,
    rectangles,
    gaps
):

    report_file = folder / "report.txt"

    width = image.shape[1]
    height = image.shape[0]

    with open(report_file, "w", encoding="utf-8") as report:

        write_heading(report, "Receipt Inspector Report")

        report.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        write_section(report, "File")

        report.write(f"Name: {file_name}\n\n")

        write_section(report, "Image")

        report.write(f"Width : {width} px\n")
        report.write(f"Height: {height} px\n\n")

        write_section(report, "Processing")

        report.write("Threshold : Otsu\n\n")

        write_section(report, "Detection")

        report.write(
            f"Receipt Candidates : {len(rectangles)}\n"
        )

        report.write(
            f"Vertical Gaps      : {len(gaps)}\n\n"
        )

        write_section(report, "Observations")

        report.write("✓ Image loaded successfully\n")
        report.write("✓ Threshold applied\n")
        report.write("✓ Receipt detector completed\n")
        report.write("✓ Gap detector completed\n")