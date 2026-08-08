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

        accepted_measurements = []

        for measurement in stats["contour_measurements"]:

            if measurement["area"] >= 10000:

                accepted_measurements.append(
                    measurement
                )

        for index, measurement in enumerate(
            accepted_measurements,
            start=1
        ):

            report.write(
                f"Contour {index}\n\n"
            )

            report.write(
                f"Area            : "
                f"{measurement['area']:,.0f} px²\n"
            )

            report.write(
                f"Width           : "
                f"{measurement['width']} px\n"
            )

            report.write(
                f"Height          : "
                f"{measurement['height']} px\n"
            )

            report.write(
                f"Bounding Area   : "
                f"{measurement['bounding_area']:,} px²\n"
            )

            report.write(
                f"Aspect Ratio    : "
                f"{measurement['aspect_ratio']:.3f}\n"
            )

            report.write(
                f"Fill Ratio      : "
                f"{measurement['fill_ratio']:.3f}\n"
            )

            report.write(
                f"Perimeter       : "
                f"{measurement['perimeter']:,.0f} px\n"
            )

            report.write(
                f"Rectangularity  : "
                f"{measurement['rectangularity']:.3f}\n\n"
            )

        if not accepted_measurements:

            report.write(
                "No accepted contours to measure.\n\n"
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