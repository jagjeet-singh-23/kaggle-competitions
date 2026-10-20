"""Phase B: sed_train.py unchanged, trained on the Perch-teacher targets (--labels B).

The code dataset carries sed_train.py; this only sets its arguments, since a Kaggle
script kernel cannot pass any.
"""
import os
import runpy
import sys

for dirpath, _, files in os.walk("/kaggle/input"):
    if "sed_train.py" in files:
        break
else:
    raise FileNotFoundError("sed_train.py not found under /kaggle/input")

sys.argv = ["sed_train.py", "--labels", "B"]
runpy.run_path(f"{dirpath}/sed_train.py", run_name="__main__")
