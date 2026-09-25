# Including libraries
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Candidate directories that might contain the robot model modules.
CANDIDATE_DIRS = [
    ROOT_DIR,
    os.path.join(ROOT_DIR, "robots"),
]

for path in CANDIDATE_DIRS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

def pytest_report_header(config):
    return f"conftest: added to sys.path -> {[p for p in CANDIDATE_DIRS if os.path.isdir(p)]}"