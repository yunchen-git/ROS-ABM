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