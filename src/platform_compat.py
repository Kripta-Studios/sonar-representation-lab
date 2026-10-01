"""Private Python 3.12 workaround for this host's failing WMI identity query.

CPython platform already supports OS/architecture environment fallbacks when
WMI is unavailable. Use that path locally; do not change Windows services.
"""

import platform
import sys

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

if sys.platform == "win32":

    def unavailable_wmi(*args, **kwargs):
        raise OSError("Project-local fallback: host WMI identity query failed")

    platform._wmi_query = unavailable_wmi
