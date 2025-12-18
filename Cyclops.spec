# -*- mode: python ; coding: utf-8 -*-
"""
Cyclops - PyInstaller Build Spec
GitHub Actions için yapılandırma
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# MediaPipe ve OpenCV verilerini topla
mediapipe_datas = collect_data_files('mediapipe')
# cv2_datas = collect_data_files('cv2') # Headless surumde cakismayi onlemek icin kapattik

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('calibration.json', '.'),
        ('settings.json', '.'),
    ] + mediapipe_datas,
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.face_mesh',
        'cv2',
        'numpy',
        'PyQt6',
        'pyautogui',
        'pynput',
    ] + collect_submodules('mediapipe'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'tkinter', 'torch', 'tensorflow', 'matplotlib', 'scipy', 'sklearn',
        'cv2.cv2', 'opencv-python' 
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platforma göre derleme ayarları
if sys.platform == 'darwin':
    # macOS .app
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Cyclops',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=True,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='Cyclops',
    )
    app = BUNDLE(
        coll,
        name='Cyclops.app',
        icon='assets/icons/logo.png',
        bundle_identifier='com.mehmetaltunel.cyclops',
        info_plist={
            'NSCameraUsageDescription': 'Göz takibi için kamera erişimi gerekli.',
            'NSHighResolutionCapable': True
        },
    )
else:
    # Windows .exe (Tek dosya)
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='Cyclops',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/icons/logo.png',
    )
