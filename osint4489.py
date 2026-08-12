#!/usr/bin/env python3
"""
4489 OSINT Tool v1 — Silnik Geolokalizacji AI
Polska wersja z obsługą GUI i CLI

Użycie:
  python osint4489.py                             # Uruchom interfejs GUI
  python osint4489.py search --image photo.jpg    # Wyszukiwanie CLI
  python osint4489.py index --lat .. --lon ..     # Tworzenie bazy CLI
  python osint4489.py list-indexes                # Wyświetlenie lokalnych baz
  python osint4489.py match --image1 a --image2 b # Dopasowanie MASt3R 3D
  python osint4489.py hub list                    # Przeglądanie bazy społeczności
"""

import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Import głównej logiki aplikacji z test_super
import test_super

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_super.run_cli()
    else:
        # Uruchomienie GUI
        for d in [test_super.DATA_DIR, test_super.MEGALOC_PARTS_DIR, test_super.INDEXES_DIR]:
            os.makedirs(d, exist_ok=True)
        available = test_super.scan_indexes()
        if available:
            test_super.load_index(available[-1]["index_id"])
        
        import tkinter as tk
        root = tk.Tk()
        app = test_super.StreetViewMatcherGUI(root)
        root.mainloop()
