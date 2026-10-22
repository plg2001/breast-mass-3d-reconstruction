"""
Forwarder script for Ling Shao (2007) Partial Volume Compensated 3D Mass Shape Reconstruction.
Redirects execution to '01_Ricostruzione_Lesione_3D/lesion_pvc_reconstruction.py'.
"""
import os
import sys
from pathlib import Path

import os
import sys
import runpy
from pathlib import Path

TARGET_SCRIPT = Path(__file__).resolve().parent / "01_Ricostruzione_Lesione_3D" / "lesion_pvc_reconstruction.py"

if __name__ == "__main__":
    os.chdir(TARGET_SCRIPT.parent)
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")
