#!/usr/bin/env python3
"""
4489 OSINT Tool v1 (dawniej NO NAME v2) — Launcher
"""
import sys
import osint4489

if __name__ == "__main__":
    if len(sys.argv) > 1:
        osint4489.test_super.run_cli()
    else:
        for d in [osint4489.test_super.DATA_DIR, osint4489.test_super.MEGALOC_PARTS_DIR, osint4489.test_super.INDEXES_DIR]:
            os.makedirs(d, exist_ok=True)
        available = osint4489.test_super.scan_indexes()
        if available:
            osint4489.test_super.load_index(available[-1]["index_id"])
        import tkinter as tk
        root = tk.Tk()
        app = osint4489.test_super.StreetViewMatcherGUI(root)
        root.mainloop()
