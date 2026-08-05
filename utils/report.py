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
    gaps,
    timings,
    stats
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

        report.write(
            f"Threshold : {stats['threshold']}\n\n"
        )

        write_section(report, "Detection")

        report.write(
            f"Contours Found      : {stats['contours_found']}\n"
        )

        report.write(
            f"Accepted Contours   : {stats['accepted_contours']}\n"
        )

        report.write(
            f"Rejected Contours   : {stats['rejected_contours']}\n\n"
        )

        report.write(
            f"Receipt Candidates  : {len(rectangles)}\n"
        )

        report.write(
            f"Vertical Gaps       : {len(gaps)}\n\n"
        )

        write_section(report, "Contour Measurements")

        report.write(
            f"Largest Contour     : {stats['largest_contour']:,} px²\n"
        )

        report.write(
            f"Smallest Contour    : {stats['smallest_contour']:,} px²\n"
        )

        report.write(
            f"Average Contour     : {stats['average_contour']:,} px²\n\n"
        )

        write_section(report, "Performance")

        for stage, seconds in timings.items():

            report.write(
                f"{stage:<20} {seconds:.3f} s\n"
            )

        report.write("\n")

        write_section(report, "Observations")

        report.write("✓ Image loaded successfully\n")
        report.write("✓ Threshold applied\n")
        report.write("✓ Receipt detector completed\n")
        report.write("✓ Gap detector completed\n")