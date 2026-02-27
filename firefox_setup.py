import shutil
import os
import sys
import json
import zipfile
from playwright.sync_api import sync_playwright

from config import (
    FIREFOX_ADBLOCK_MASTER_PROFILE,
    FIREFOX_NOADBLOCK_MASTER_PROFILE,
    UBLOCK_FIREFOX_XPI_PATH  # path to the raw .xpi file, e.g. extensions/ublock_origin.xpi
)

def get_extension_id(xpi_path):
    """Reads the extension ID from the manifest inside the .xpi file."""
    with zipfile.ZipFile(xpi_path, 'r') as z:
        with z.open('manifest.json') as f:
            manifest = json.load(f)
    # The ID is in browser_specific_settings for modern extensions
    try:
        return manifest['browser_specific_settings']['gecko']['id']
    except KeyError:
        return manifest['applications']['gecko']['id']

def install_xpi_into_profile(profile_path, xpi_path):
    """
    Copies the .xpi into the profile's extensions folder.
    Firefox will pick it up automatically on next launch.
    """
    ext_id = get_extension_id(xpi_path)
    extensions_dir = os.path.join(profile_path, "extensions")
    os.makedirs(extensions_dir, exist_ok=True)
    dest = os.path.join(extensions_dir, f"{ext_id}.xpi")
    shutil.copy2(xpi_path, dest)
    print(f"Copied extension '{ext_id}' into profile.")

def setup_profile(p, profile_path, use_adblock: bool):
    label = "ADBLOCK" if use_adblock else "NO-ADBLOCK"
    print(f"\n--- Setting up {label} profile ---")

    if os.path.exists(profile_path):
        answer = input(f"Profile already exists at '{profile_path}'. Overwrite? [y/n] ")
        if answer.lower() != 'y':
            print("Skipping.")
            return
        shutil.rmtree(profile_path)

    # If adblock, install the xpi into the profile folder BEFORE launching,
    # so Firefox finds it on first boot
    if use_adblock:
        print("Pre-installing uBlock Origin into profile folder...")
        install_xpi_into_profile(profile_path, UBLOCK_FIREFOX_XPI_PATH)

    context = p.firefox.launch_persistent_context(
        profile_path,
        headless=False,
        no_viewport=True
    )

    print(f"Firefox is open with the {label} profile.")
    if use_adblock:
        print(">>> Verify uBlock Origin is active in the extensions menu.")
    print(">>> Dismiss any first-run popups, then CLOSE THE BROWSER WINDOW to continue.")

    context.wait_for_event("close", timeout=0)
    print(f"{label} profile saved at '{profile_path}'.")

def main():
    if not os.path.exists(UBLOCK_FIREFOX_XPI_PATH):
        print(f"[ERROR] uBlock .xpi not found at '{UBLOCK_FIREFOX_XPI_PATH}'.")
        print("Download it from https://addons.mozilla.org/en-US/firefox/addon/ublock-origin/")
        sys.exit(1)

    with sync_playwright() as p:
        setup_profile(p, profile_path=FIREFOX_NOADBLOCK_MASTER_PROFILE, use_adblock=False)
        setup_profile(p, profile_path=FIREFOX_ADBLOCK_MASTER_PROFILE, use_adblock=True)

    print("\nAll profiles set up. You can now run the main experiment script.")

if __name__ == "__main__":
    main()