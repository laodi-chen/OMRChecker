"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

import argparse
import sys
from pathlib import Path

from src.entry import entry_point
from src.logger import logger


def _resolve_path(path_value: str):
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path

    candidates = [Path.cwd() / path]
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / path)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / path)
    else:
        candidates.append(Path(__file__).resolve().parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_cell_detection(args):
    image_path = _resolve_path(args["cell"]) or Path(args["cell"])
    weights_path = _resolve_path(args["cell_weights"]) or Path(args["cell_weights"])

    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: '{image_path}'")
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights path does not exist: '{weights_path}'")

    try:
        from infer_count import infer_count
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for ONNX inference. Please install 'onnxruntime'."
        ) from exc

    count = infer_count(
        str(weights_path),
        str(image_path),
        conf=args["cell_conf"],
        iou=args["cell_iou"],
        save_vis=not args["cell_no_vis"],
    )
    logger.info(f"Detected cell count: {count}")


def parse_args():
    # construct the argument parse and parse the arguments
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "-i",
        "--inputDir",
        default=["inputs"],
        # https://docs.python.org/3/library/argparse.html#nargs
        nargs="*",
        required=False,
        type=str,
        dest="input_paths",
        help="Specify an input directory.",
    )

    argparser.add_argument(
        "-d",
        "--debug",
        required=False,
        dest="debug",
        action="store_false",
        help="Enables debugging mode for showing detailed errors",
    )

    argparser.add_argument(
        "-o",
        "--outputDir",
        default="outputs",
        required=False,
        dest="output_dir",
        help="Specify an output directory.",
    )

    argparser.add_argument(
        "-a",
        "--autoAlign",
        required=False,
        dest="autoAlign",
        action="store_true",
        help="(experimental) Enables automatic template alignment - \
        use if the scans show slight misalignments.",
    )

    argparser.add_argument(
        "-l",
        "--setLayout",
        required=False,
        dest="setLayout",
        action="store_true",
        help="Set up OMR template layout - modify your json file and \
        run again until the template is set.",
    )
    argparser.add_argument(
        "--cell",
        required=False,
        dest="cell",
        default=None,
        help="Run YOLO cell detection for a single image path.",
    )
    argparser.add_argument(
        "--cell-weights",
        required=False,
        dest="cell_weights",
        default="best.onnx",
        help="Path to YOLO ONNX weights file used by --cell (default: best.onnx).",
    )
    argparser.add_argument(
        "--cell-conf",
        required=False,
        dest="cell_conf",
        default=0.2,
        type=float,
        help="Confidence threshold for --cell YOLO inference.",
    )
    argparser.add_argument(
        "--cell-iou",
        required=False,
        dest="cell_iou",
        default=0.25,
        type=float,
        help="IoU threshold for --cell YOLO inference.",
    )
    argparser.add_argument(
        "--cell-no-vis",
        required=False,
        dest="cell_no_vis",
        action="store_true",
        help="Disable saving YOLO visualization output for --cell.",
    )

    (
        args,
        unknown,
    ) = argparser.parse_known_args()

    args = vars(args)

    if len(unknown) > 0:
        logger.warning(f"\nError: Unknown arguments: {unknown}", unknown)
        argparser.print_help()
        exit(11)
    return args


def entry_point_for_args(args):
    if args["cell"]:
        run_cell_detection(args)
        return

    if args["debug"] is True:
        # Disable tracebacks
        sys.tracebacklimit = 0
    for root in args["input_paths"]:
        entry_point(
            Path(root),
            args,
        )


if __name__ == "__main__":
    args = parse_args()
    entry_point_for_args(args)
