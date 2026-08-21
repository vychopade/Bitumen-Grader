# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BitumenGrader.

From the repo root:

    pip install -r requirements-dev.txt
    pyinstaller BitumenGrader.spec

Output: dist/BitumenGrader/ (and BitumenGrader.app on macOS).
Do not bundle models/ or BitumenImagesFlotation/; saved models go in the
OS application-data folder (see app.paths.MODELS_DIR).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.building.api import COLLECT, EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH)
ICON_PATH = ROOT / "assets" / "logo.png"
ICON = str(ICON_PATH) if ICON_PATH.exists() else None

datas = [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "USER_GUIDE.txt"), "."),
]
binaries = []
hiddenimports = [
    "matplotlib.backends.backend_qtagg",
    "openpyxl",
    "torchvision.models.resnet",
    "torchvision.models.vgg",
]

for package in ("torch", "torchvision", "PyQt6", "matplotlib"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

datas += collect_data_files("pandas")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "notebook", "pytest", "sklearn", "tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BitumenGrader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BitumenGrader",
)

if sys.platform == "darwin":
    from PyInstaller.building.osx import BUNDLE

    app = BUNDLE(
        coll,
        name="BitumenGrader.app",
        icon=ICON,
        bundle_identifier="local.bitumengrader",
        info_plist={
            "CFBundleName": "BitumenGrader",
            "CFBundleDisplayName": "BitumenGrader",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
