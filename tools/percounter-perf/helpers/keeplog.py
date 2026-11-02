"""Run a script or module with the simpler Host log bound to $KEEP_HOST_LOG_DIR.

A capture with an output prefix otherwise spools the Host log into a transient
directory removed at exit, taking the [STRACE] device_wall span with it.
usage: python keeplog.py <script.py | -m module> [args...]
"""
import os
import runpy
import sys

from simpler import task_interface as ti

d = os.environ["KEEP_HOST_LOG_DIR"]
os.makedirs(d, exist_ok=True)
ti._native_set_host_log_directory(d)
if sys.argv[1] == "-m":
    sys.argv = [sys.argv[2]] + sys.argv[3:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
