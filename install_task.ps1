$py = "$PWD\.venv\Scripts\python.exe"
$main = "$PWD\app_modular.py"
schtasks /Create /SC DAILY /TN "GoldenLitigatorDaily" /TR "`"$py`" `"$main`"" /ST 02:30 /F
