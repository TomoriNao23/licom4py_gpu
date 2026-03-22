#!/usr/bin/env python3
"""Helper: read namelist and print NX PX PY NDEV NP on one line."""
import configparser, sys

c = configparser.ConfigParser()
c.read(sys.argv[1] if len(sys.argv) > 1 else "src/licom/namelist")

nx   = int(c["grid"]["nx"])
px   = int(c["gpu_mesh"]["px"])
py   = int(c["gpu_mesh"]["py"])
pdev = int(c["gpu_mesh"]["pdev"])
np   = pdev * px * py
np_field = 6 * px * py

print(f"{nx} {px} {py} {pdev} {np} {np_field}")
