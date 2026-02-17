################################################################################
# File: CountDirectory.py
# Purpose:
#   Utility for run folder naming:
#   scans a save directory for numeric folder names (e.g., "1.0", "2.0") and
#   returns the next available number to use as a new run ID.
#
# Key public API:
#   - newDirect(path) -> float
#
# How it fits in the codebase:
#   Used by Model_run.py / Model_initialParameters.py to create new run folders
#   without overwriting prior results.
#
# Inputs/Outputs:
#   - Reads directory listings from disk (os.listdir).
#   - Returns the next numeric run label (float).
#
# Notes / gotchas:
#   - Non-numeric entries are ignored.
#   - Raises FileNotFoundError if the path does not exist.
################################################################################
import os
def newDirect(path):
    """
    This script opens the specified save path and finds the highest folder number
    It then returns the next highest number as a name for the currently running simulation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")
    
    files = os.listdir(path)
    numFiles = []
    
    for file in files:
        try:
            numFiles.append(float(file))
        except ValueError:
            pass
    
    if numFiles:
        k = max(numFiles)
    else:
        k = 0
    
    return k + 1.0