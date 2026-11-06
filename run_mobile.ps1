$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Project venv not found. Create it and install dependencies from requirements.txt first."
}

$route = Get-NetRoute -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue |
    Sort-Object RouteMetric |
    Select-Object -First 1
$lanIp = if ($route) {
    Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $route.InterfaceIndex -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "169.254.*" } |
        Select-Object -First 1 -ExpandProperty IPAddress
}

if ($lanIp) {
    Write-Host "On this PC: http://127.0.0.1:8501"
    Write-Host "On a phone on the same trusted Wi-Fi: http://${lanIp}:8501"
    Write-Host "Keep this terminal open while using the app. Do not share this URL outside your trusted network."
} else {
    Write-Warning "Could not detect a Wi-Fi/LAN IPv4 address. The app will still start on this PC."
}

& $python -m streamlit run app.py `
    --server.headless true `
    --server.address 0.0.0.0 `
    --server.port 8501 `
    --server.enableCORS true `
    --server.enableXsrfProtection true
