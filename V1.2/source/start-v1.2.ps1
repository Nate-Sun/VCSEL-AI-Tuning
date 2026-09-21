[CmdletBinding()]
param(
    [ValidateSet("FAKE", "READ_ONLY", "LIVE_TUNING")]
    [string]$Mode = "FAKE",

    [string]$ConfigPath,

    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [switch]$NoBrowser,

    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# start-v1.1.ps1 remains the compatibility implementation because API/schema
# identifiers stay at v1.1. This wrapper is the public V1.2 entry point.
$Implementation = Join-Path $PSScriptRoot "start-v1.1.ps1"
if (-not (Test-Path -LiteralPath $Implementation -PathType Leaf)) {
    throw "V1.2 launcher implementation is missing: $Implementation"
}

& $Implementation @PSBoundParameters
