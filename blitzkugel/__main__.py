"""Entry point: ``python -m blitzkugel [datei.step]``."""

from __future__ import annotations

import argparse
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    """Start the GUI."""
    parser = argparse.ArgumentParser(prog="blitzkugel", description=__doc__)
    parser.add_argument("step_file", nargs="?", help="STEP-Datei, die beim Start geladen wird")
    parser.add_argument("-v", "--verbose", action="store_true", help="ausführliches Logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    os.environ.setdefault("QT_API", "pyqt6")  # pyvistaqt/qtpy must use PyQt6

    from PyQt6.QtWidgets import QApplication  # noqa: PLC0415

    from .gui.main_window import MainWindow  # noqa: PLC0415

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Blitzkugel")
    window = MainWindow()
    window.show()
    if args.step_file:
        window.load_step_file(args.step_file)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
