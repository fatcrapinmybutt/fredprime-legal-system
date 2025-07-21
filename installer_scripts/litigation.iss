[Setup]
AppName=FRED Supreme Litigation OS
AppVersion=1.0
DefaultDirName={autopf}\FRED_SUPREME_LITIGATION_OS
DefaultGroupName=FRED Supreme Litigation OS
OutputDir=output\installer_scripts
OutputBaseFilename=litigation_launcher_setup
SetupIconFile=docs\logo.ico
Compression=lzma2
SolidCompression=yes

[Files]
Source: "dist\litigation_launcher.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "docs\*"; DestDir: "{app}\docs"; Flags: recursesubdirs
Source: "binder\*"; DestDir: "{app}\binder"; Flags: recursesubdirs
Source: "config\*"; DestDir: "{app}\config"; Flags: recursesubdirs
Source: "contradictions\*"; DestDir: "{app}\contradictions"; Flags: recursesubdirs
Source: "entity_trace\*"; DestDir: "{app}\entity_trace"; Flags: recursesubdirs
Source: "foia\*"; DestDir: "{app}\foia"; Flags: recursesubdirs
Source: "gui\*"; DestDir: "{app}\gui"; Flags: recursesubdirs
Source: "mifile\*"; DestDir: "{app}\mifile"; Flags: recursesubdirs
Source: "modules\*"; DestDir: "{app}\modules"; Flags: recursesubdirs
Source: "motions\*"; DestDir: "{app}\motions"; Flags: recursesubdirs
Source: "notices\*"; DestDir: "{app}\notices"; Flags: recursesubdirs
Source: "scanner\*"; DestDir: "{app}\scanner"; Flags: recursesubdirs
Source: "scheduling\*"; DestDir: "{app}\scheduling"; Flags: recursesubdirs
Source: "src\*"; DestDir: "{app}\src"; Flags: recursesubdirs
Source: "timeline\*"; DestDir: "{app}\timeline"; Flags: recursesubdirs
Source: "warboard\*"; DestDir: "{app}\warboard"; Flags: recursesubdirs
Source: "output\*"; DestDir: "{app}\output"; Flags: recursesubdirs
Source: "profile\*"; DestDir: "{app}\profile"; Flags: recursesubdirs
Source: "ai\*"; DestDir: "{app}\ai"; Flags: recursesubdirs
Source: "api\*"; DestDir: "{app}\api"; Flags: recursesubdirs
Source: "backup\*"; DestDir: "{app}\backup"; Flags: recursesubdirs
Source: "mobile\*"; DestDir: "{app}\mobile"; Flags: recursesubdirs
Source: "events\*"; DestDir: "{app}\events"; Flags: recursesubdirs
Source: "export\*"; DestDir: "{app}\export"; Flags: recursesubdirs
Source: "import_export\*"; DestDir: "{app}\import_export"; Flags: recursesubdirs
Source: "forensic\*"; DestDir: "{app}\forensic"; Flags: recursesubdirs
Source: "sim\*"; DestDir: "{app}\sim"; Flags: recursesubdirs

[Icons]
Name: "{group}\FRED Supreme Litigation OS"; Filename: "{app}\litigation_launcher.exe"
Name: "{commondesktop}\FRED Supreme Litigation OS"; Filename: "{app}\litigation_launcher.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\litigation_launcher.exe"; Description: "Launch FRED Supreme Litigation OS"; Flags: nowait postinstall skipifsilent

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"
