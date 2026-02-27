import json
import os
import sys

CONFIG_FILE_PATH = 'config.json'
SITES_FILE_PATH = 'sites.json'

def find_dll():
    winget_root = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"
    )
    for root, dirs, files in os.walk(winget_root):
        if "LibreHardwareMonitorLib.dll" in files:
            return os.path.join(root, "LibreHardwareMonitorLib.dll")
    raise FileNotFoundError("LibreHardwareMonitorLib.dll not found.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UBLOCK_LITE_CHROME_PATH = os.path.join(BASE_DIR, "extensions", "ublock_chrome")
UBLOCK_EDGE_PATH = os.path.join(BASE_DIR, "extensions", "ublock_edge")

USER_DATA_DIR_CHROME = os.path.join(BASE_DIR, "playwright_user_data_chrome")
USER_DATA_DIR_EDGE = os.path.join(BASE_DIR, "playwright_user_data_edge")

UBLOCK_FIREFOX_XPI_PATH = os.path.join(BASE_DIR, "extensions", "ublock_firefox", "ublock_origin-1.69.0.xpi") 

FIREFOX_ADBLOCK_MASTER_PROFILE = os.path.join(BASE_DIR, "firefox_profiles", "adblock_master")
FIREFOX_NOADBLOCK_MASTER_PROFILE = os.path.join(BASE_DIR, "firefox_profiles", "noadblock_master")
USER_DATA_DIR_FIREFOX_ADBLOCK = os.path.join(BASE_DIR, "playwright_user_data_firefox_adblock")
USER_DATA_DIR_FIREFOX_NOADBLOCK = os.path.join(BASE_DIR, "playwright_user_data_firefox_noadblock")

OUTPUT_FILE = os.path.join(BASE_DIR, "output.csv")
DLL_PATH = find_dll()

try:
    with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
        _config_data = json.load(f)
    with open(SITES_FILE_PATH, 'r', encoding='utf-8') as f:
        SITES_TO_TEST = json.load(f)
except FileNotFoundError as e:
    print(f"Error: A required configuration file was not found.")
    print(f"Please ensure '{CONFIG_FILE_PATH}' and '{SITES_FILE_PATH}' exist in the same directory as the script.")
    print(f"Details: {e}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: A configuration file is not valid JSON.")
    print(f"Please check the syntax of the file that caused the error.")
    print(f"Details: {e}")
    sys.exit(1)

try:
    _measurement_config = _config_data['measurement']
    _analysis_config = _config_data['analysis']

    # Durations
    WARMUP_SECONDS = _measurement_config["warmup_seconds"]
    DURATION_SECONDS = _measurement_config['duration_seconds']
    COOLDOWN_SECONDS = _measurement_config['cooldown_seconds']
    WAIT_MEAN = _measurement_config['wait_between_actions_mean_s']
    WAIT_STD = _measurement_config['wait_between_actions_std_s']

    # Analysis
    POWER_COLUMN = _analysis_config['power_column_name']
    AD_BLOCKLIST_FILE = _analysis_config["blocklist_file"]


except KeyError as e:
    print(f"Error: A required key is missing from your '{CONFIG_FILE_PATH}' file.")
    print(f"The missing key is: {e}")
    print("Please ensure your configuration is complete.")
    sys.exit(1)

print("Configuration loaded successfully.")