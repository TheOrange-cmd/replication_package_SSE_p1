
# Replication Package

## Requirements

- Windows (Linux and macOS are not supported)
- Conda installed (mamba, uv, poetry, or a plain venv also work)
- Administrator access

**VS Code users:** use a CMD terminal, not PowerShell, when working with conda.

---

## Setup

### 1. Clone the repository and dependencies

```bash
git clone https://github.com/TheOrange-cmd/replication_package_SSE_p1.git
cd replication_package_SSE_p1
git clone https://github.com/effeect/LibreHardwareMonitorCLI.git
```

### 2. Install LibreHardwareMonitor

```bash
winget install LibreHardwareMonitor.LibreHardwareMonitor --source winget
```

### 3. Create the Python environment

```bash
conda create -n SSE python pythonnet numpy pandas playwright playwright-stealth matplotlib seaborn numba scikit-learn
conda activate SSE
python -m playwright install
```

### 4. Enable Windows long path support

Some Firefox profile paths exceed Windows' default path length limit. Run this once in an admin PowerShell:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 5. Download browser extensions

**Chrome — uBlock Origin Lite**

Chrome no longer supports uBlock Origin; uBlock Origin Lite is used instead.

1. Go to `https://crxviewer.com/?crx=https://chrome.google.com/webstore/detail/ddkjiahejlhfcafbddmgiahcphecmpfh`
2. Click *Download as zip*, unzip, and place the folder at `extensions/ublock_chrome/`

**Edge — uBlock Origin**

1. Find the extension page, e.g. `https://microsoftedge.microsoft.com/addons/detail/ublock-origin/odfafepnkmbhccpbejgmiehpchacaeak`
2. Copy the ID at the end of the URL (e.g. `odfafepnkmbhccpbejgmiehpchacaeak`)
3. Download the `.crx` file by visiting:
```
https://edge.microsoft.com/extensionwebstorebase/v1/crx?response=redirect&x=id%3D[EXTENSION_ID]%26installsource%3Dondemand%26uc
```
4. Open the downloaded file with 7-Zip as an archive and extract the contents to `extensions/ublock_edge/`

**Firefox — uBlock Origin**

1. In Firefox, navigate to `https://addons.mozilla.org/en-US/firefox/addon/ublock-origin`
2. Right-click *Add to Firefox* and choose *Save link as*
3. Save the file to `extensions/ublock_firefox/`

### 6. Set up Firefox profiles

Playwright uses Firefox Nightly and requires pre-configured profiles for the adblock and no-adblock conditions. Create them by running:

```bash
python firefox_setup.py
```

---

## Running the experiment

The experiment script requires administrator access to read hardware sensors.

**Before running**, reduce measurement variability by preparing the machine:

- Close all non-essential applications and services
- Use a wired Ethernet connection
- Disable automatic updates and notifications
- Set a fixed screen brightness; disable auto-brightness
- Set the power plan to *High Performance*
- Run in a temperature-stable environment

**Then run:**

```bash
conda activate SSE
cd [path_to_project]
python browse.py
```

---

## Running the analysis

```bash
# Pilot analysis only - to investigate sample size for full experiment
python analyze_pilot_study.py

# Full analysis
python analysis.py