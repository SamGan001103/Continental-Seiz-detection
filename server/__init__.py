"""Part one of the system: everything that runs on the hospital's compute
machine, and nothing that runs on a clinical workstation.

Every model in this project runs from here. `prepare_study.py` is the entry
point — it drives the detectors and writes a study bundle, which is then
copied to the review workstations. The review application in `gui/` reads
those bundles and runs no model at all; `tests/test_large_file_open.py`
asserts it cannot.

Keeping these three modules in their own package is not tidiness. The claim
the system makes is that heavy computation happens on the server, and a
reader should be able to see where that boundary falls without tracing
imports.
"""
