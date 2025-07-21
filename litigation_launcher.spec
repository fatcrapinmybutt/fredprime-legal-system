# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui/main_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('docs/*', 'docs'),
        ('binder/*', 'binder'),
        ('config/*', 'config'),
        ('contradictions/*', 'contradictions'),
        ('entity_trace/*', 'entity_trace'),
        ('foia/*', 'foia'),
        ('gui/*', 'gui'),
        ('mifile/*', 'mifile'),
        ('modules/*', 'modules'),
        ('motions/*', 'motions'),
        ('notices/*', 'notices'),
        ('scanner/*', 'scanner'),
        ('scheduling/*', 'scheduling'),
        ('src/*', 'src'),
        ('timeline/*', 'timeline'),
        ('warboard/*', 'warboard'),
        ('output/*', 'output'),
        ('tests/*', 'tests'),
        ('profile/*', 'profile'),
        ('ai/*', 'ai'),
        ('api/*', 'api'),
        ('backup/*', 'backup'),
        ('mobile/*', 'mobile'),
        ('events/*', 'events'),
        ('export/*', 'export'),
        ('import_export/*', 'import_export'),
        ('forensic/*', 'forensic'),
        ('sim/*', 'sim')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='litigation_launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set True for CLI debug
    icon='docs/logo.ico'  # Optional: add your .ico here
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='litigation_launcher_dist'
)
