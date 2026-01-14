# golden_god_mode_bootstrap.spec
# PyInstaller spec for building a portable EXE
block_cipher = None

a = Analysis(["golden_god_mode_bootstrap.py"], pathex=["."])
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="golden_god_mode_bootstrap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="golden_god_mode_bootstrap",
)
