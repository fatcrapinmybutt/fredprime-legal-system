param(
  [Parameter(Mandatory=$true)][string]$Root,
  [string]$Tag="__DONE__"
)
Get-ChildItem -Path $Root -Recurse -File | Where-Object { $_.BaseName -like "*$Tag" } | ForEach-Object {
  $new = $_.FullName -replace [Regex]::Escape($Tag), ""
  try { Rename-Item -LiteralPath $_.FullName -NewName (Split-Path $new -Leaf) -ErrorAction Stop } catch {}
}
