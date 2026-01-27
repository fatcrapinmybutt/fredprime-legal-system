$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "python"
& $python (Join-Path $root "OUT\code\mcl_plane_suite.py") --in (Join-Path $root "IN") --out (Join-Path $root "OUT") --mode converge --adversarial true --web false
