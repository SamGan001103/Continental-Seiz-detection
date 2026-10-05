"""Entry point for the seizure-review GUI.

Launch from repo root:
    python -m gui.main                 # empty window
    python -m gui.main <path/to.edf>   # auto-load a file
"""
import os
import sys
from PyQt5 import QtWidgets

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gui.app import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith('.edf'):
        w.load_edf(os.path.abspath(sys.argv[1]))
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
