'''
Main script for the browsing energy use task.

Pre run checklist:

Prepare the Test Machine:

    Close all non-essential applications and services.
    Plug into a wired Ethernet connection.
    Disable automatic updates and notifications.
    Set a fixed screen brightness and resolution. Disable auto-brightness.
    Select the "High Performance" power plan.
    Position the Machine: Place it in a temperature-stable environment.

'''

# General imports
import time
import csv
import threading
import random 
from datetime import datetime
import shutil 
import os
import ctypes
import sys
import math

# Playwright
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth

# Local utils
from hardware_utils import warmup_cpu, warmup_gpu, HardwareMonitor

# Load config globals
from config import (
    # Chrome
    USER_DATA_DIR_CHROME,
    UBLOCK_LITE_CHROME_PATH,
    # Edge
    USER_DATA_DIR_EDGE,
    UBLOCK_EDGE_PATH,
    # Firefox
    USER_DATA_DIR_FIREFOX_ADBLOCK, 
    USER_DATA_DIR_FIREFOX_NOADBLOCK,
    FIREFOX_ADBLOCK_MASTER_PROFILE,
    FIREFOX_NOADBLOCK_MASTER_PROFILE,
    # Dll for librehardwaremon
    DLL_PATH, 
    # File for results
    OUTPUT_FILE,
    # Timing parameters
    WARMUP_SECONDS,
    DURATION_SECONDS, 
    COOLDOWN_SECONDS, 
    SITES_TO_TEST,
    WAIT_MEAN,
    WAIT_STD,
    # List of known ad URLs for data estimation
    AD_BLOCKLIST_FILE
)

class NetworkTrafficMonitor:
    """
    A thread-safe class to monitor and classify network traffic on a Playwright page.
    It uses a domain blocklist to identify ad-related traffic.
    """
    def __init__(self, blocklist_path):
        self._lock = threading.Lock()
        self.blocklist = self._load_blocklist(blocklist_path)
        
        # Metrics
        self.total_requests = 0
        self.total_bytes = 0
        self.ad_requests = 0
        self.ad_bytes = 0
        self.blocked_by_client_requests = 0
        
        # Page object
        self._page = None

    def _load_blocklist(self, blocklist_path):
        """Loads the ad domain blocklist from a file into a set for fast lookups."""
        if not os.path.exists(blocklist_path):
            print(f"[WARNING] Ad blocklist file not found at '{blocklist_path}'. Ad traffic will not be classified.")
            # Create an empty file to avoid future errors
            with open(blocklist_path, 'w') as f:
                f.write("# Please add ad-serving domains here, one per line.\n")
                f.write("# Example: doubleclick.net\n")
            return set()
        
        with open(blocklist_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines/comments
            return {line[10:].strip() for line in f if line[10:].strip() and not line.startswith('#')}

    def _is_ad_domain(self, url):
        """Checks if the domain of a URL is in our blocklist."""
        try:
            domain = url.split('/')[2]
            for ad_domain in self.blocklist:
                if domain.endswith(ad_domain):
                    return True
        except IndexError:
            return False
        return False

    def _handle_response(self, response):
        """Callback for the 'response' event to log and classify traffic."""
        try:
            # This is much more efficient than response.body() as it doesn't load the
            # entire response content into Python's memory.
            sizes = response.request.sizes()
            body_size = sizes.get('responseBodySize', 0)
            
            if body_size > 0:
                is_ad = self._is_ad_domain(response.url)
                
                with self._lock:
                    self.total_requests += 1
                    self.total_bytes += body_size
                    if is_ad:
                        self.ad_requests += 1
                        self.ad_bytes += body_size
        except Exception:
            # This can happen if the request is cancelled or fails before size is determined.
            # We can safely ignore it.
            pass

    def _handle_request_failed(self, request):
        """Callback for the 'requestfailed' event to count requests blocked by the ad blocker."""
        try:
            if request.failure and request.failure == 'net::ERR_BLOCKED_BY_CLIENT':
                with self._lock:
                    self.blocked_by_client_requests += 1
        except Exception as e:
            # Log if an unexpected error happens, but don't crash the script
            print(f"NETWORK_MONITOR: Unexpected error in _handle_request_failed: {e}")


    def start(self, page):
        """Starts monitoring by attaching event listeners to the page."""
        self._page = page
        self._page.on("response", self._handle_response)
        self._page.on("requestfailed", self._handle_request_failed)
        print("NETWORK_MONITOR: Started listening to network events.")

    def stop(self):
        """Stops monitoring by removing event listeners."""
        if self._page:
            try:
                self._page.remove_listener("response", self._handle_response)
                self._page.remove_listener("requestfailed", self._handle_request_failed)
                print("NETWORK_MONITOR: Stopped listening to network events.")
            except Exception as e:
                print(f"NETWORK_MONITOR: Error while stopping listeners: {e}")

    def get_current_stats(self):
        """Returns a thread-safe snapshot of the current network stats."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_bytes": self.total_bytes,
                "ad_requests": self.ad_requests,
                "ad_bytes": self.ad_bytes,
                "blocked_by_client_requests": self.blocked_by_client_requests
            }

# Define scrolling and clicking functions. Scrolling is random, clicking is specific per site, see sites.json. 

def simulate_doom_scrolling(page, duration, url):
    """
    Simulates a user scrolling down the page randomly for a given duration.
    This helps trigger lazy-loaded content like ads.

    Args:
        page: The Playwright page object to interact with.
        duration: The total number of seconds to perform the scrolling action.
    """
    print(f"BROWSER: Starting random scrolling for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:

        # Determine a random amount to scroll down.
        scroll_delta_y = random.randint(300, 900)
        
        # Perform the scroll action using the mouse wheel. 
        # Disable for youtube since users want to watch the video, not mindlessly scroll. 
        if "youtube" not in url:
            page.mouse.wheel(0, scroll_delta_y)
        
        # Determine a random wait time before the next scroll.
        wait_time = random.uniform(1.5, 4.5)
        
        # Ensure we don't wait past the total duration of the experiment.
        elapsed_time = time.time() - start_time
        remaining_time = duration - elapsed_time
        
        # If the random wait time is longer than the time we have left,
        # just sleep for whatever time is remaining.
        actual_wait = min(wait_time, remaining_time)
        
        if actual_wait > 0:
            time.sleep(actual_wait)
    
    print("BROWSER: Finished random scrolling.")

def perform_site_actions(page, actions):
    """
    Performs site-specific actions (clicks, etc.) on the given page.
    """
    if not actions:
        print("BROWSER: No actions defined for this site. Proceeding.")
        return

    print(f"BROWSER: Found {len(actions)} specific action(s) for this site.")
    for i, action in enumerate(actions):
        is_optional = action.get('is_optional', False)
        timeout = 3000 if is_optional else 15000

        try:
            print(f"BROWSER: Performing action {i+1}/{len(actions)}: {action.get('description', 'No description')} {'(Optional)' if is_optional else ''}")

            target = page
            if action.get('is_iframe'):
                iframe_selector = action.get('iframe_selector')
                if not iframe_selector:
                    print("BROWSER: Action specifies iframe but is missing 'iframe_selector'. Skipping.")
                    continue
                print(f"BROWSER: Action is inside an iframe. Locating iframe: {iframe_selector}")
                page.wait_for_selector(iframe_selector, state='visible', timeout=timeout)
                target = page.frame_locator(iframe_selector).first

            locator = None
            selector_type = action['selector_type']
            selector_value = action['selector_value']
            options = action.get('options', {})

            if selector_type == 'role':
                locator = target.get_by_role(selector_value, **options)
            elif selector_type == 'locator':
                locator = target.locator(selector_value)
            elif selector_type == 'text':
                locator = target.get_by_text(selector_value, **options)
            else:
                print(f"BROWSER: Unknown selector_type '{selector_type}'. Skipping action.")
                continue

            locator.first.click(timeout=timeout)
            print("BROWSER: Click successful.")

            wait_time = max(0.5, random.normalvariate(WAIT_MEAN, WAIT_STD))
            print(f"BROWSER: Waiting for {wait_time:.2f} seconds before next action.")
            time.sleep(wait_time)

        except Exception:
            if is_optional:
                print(f"BROWSER: Optional element not found or visible within {timeout}ms. Skipping ...")
            else:
                print("BROWSER: FAILED to perform REQUIRED action. This may affect results.")
            continue

# Define concurrent functions for hardware sensor monitoring and browser activity

def run_monitoring_task(monitor: HardwareMonitor, network_monitor: NetworkTrafficMonitor, config: dict, stop_event: threading.Event):
    """
    This function runs in a separate thread, collecting data from all discovered
    sensors and the shared network monitor, writing it to a shared CSV file.
    """
    output_file = config["output_file"]
    
    # Add new network headers to the CSV
    metadata_headers = ["timestamp", "experiment_name", "url", "adblock_enabled", "run_id"]
    network_headers = ["total_bytes", "ad_bytes", "total_requests", "ad_requests", "blocked_by_client_requests"]
    all_headers = metadata_headers + network_headers + monitor.sensor_headers

    file_exists = os.path.isfile(output_file)
    
    print(f"MONITOR: Starting monitoring. Data will be appended to '{output_file}'")
    
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(all_headers)
        
        while not stop_event.is_set():
            # Get hardware metrics
            hw_metrics = monitor.get_all_metrics()
            timestamp = datetime.now().isoformat()
            
            # Get a snapshot of network metrics from the shared monitor object
            net_stats = network_monitor.get_current_stats()
            
            metadata_values = [
                timestamp,
                config['name'],
                config['url'],
                config['use_adblock'],
                config['run_id']
            ]

            network_values = [
                net_stats["total_bytes"],
                net_stats["ad_bytes"],
                net_stats["total_requests"],
                net_stats["ad_requests"],
                net_stats["blocked_by_client_requests"]
            ]
            
            sensor_values = [f"{hw_metrics.get(header, 0.0):.4f}" for header in monitor.sensor_headers]
            
            writer.writerow(metadata_values + network_values + sensor_values)
            stop_event.wait(1)
            
    print("MONITOR: Stop signal received. Monitoring finished.")

def run_browser_task(config: dict, network_monitor: NetworkTrafficMonitor, stop_event: threading.Event):
    browser_type = config.get("browser", "chrome")
    print(f"BROWSER: Starting {browser_type.upper()} automation for experiment: '{config['name']}'")

    try:
        with sync_playwright() as p:
            try:
                if browser_type == "edge":
                    _run_edge_context(p, config, network_monitor)
                elif browser_type == "firefox":
                    _run_firefox_context(p, config, network_monitor)
                else:
                    _run_chrome_context(p, config, network_monitor)

            except Exception as e:
                print(f"BROWSER: A critical error occurred in the browser task: {e}")
            finally:
                if network_monitor:
                    network_monitor.stop()
    finally:
        print("BROWSER: Signaling monitor to stop.")
        stop_event.set()

def _launch_and_run(context, config, network_monitor):
    """Shared logic after a context is launched — navigate, scroll, etc."""
    try:
        if config["use_adblock"]:
            print("BROWSER: Waiting for ad blocker to initialize...")
            time.sleep(5)

        # Reuse the existing blank page if one was opened on launch (Firefox),
        # otherwise create a new tab (Chrome/Edge don't open one automatically).
        existing_pages = context.pages
        page = existing_pages[0] if existing_pages else context.new_page()

        network_monitor.start(page)

        stealth = Stealth()
        stealth.apply_stealth_sync(page)

        print(f"BROWSER: Navigating to {config['url']}...")
        page.goto(config['url'], timeout=90000, wait_until="domcontentloaded")

        perform_site_actions(page, config.get("actions", []))
        # Enable scrolling, note scrolling is disabed for youtbe
        simulate_doom_scrolling(page, config['duration_seconds'], config['url'])

        print("BROWSER: Task finished.")
    finally:
        context.close()
        print("BROWSER: Context closed.")

def _run_chrome_context(p, config, network_monitor):
    launch_args = ["--start-maximized"]
    if config["use_adblock"]:
        print("BROWSER: Chrome — Ad blocker (uBlock Lite) ENABLED.")
        launch_args.extend([
            f"--disable-extensions-except={UBLOCK_LITE_CHROME_PATH}",
            f"--load-extension={UBLOCK_LITE_CHROME_PATH}"
        ])
    else:
        print("BROWSER: Chrome — Ad blocker DISABLED.")

    context = p.chromium.launch_persistent_context(
        USER_DATA_DIR_CHROME,
        headless=False,
        channel="chrome",
        args=launch_args,
        no_viewport=True
    )
    _launch_and_run(context, config, network_monitor)

def _run_edge_context(p, config, network_monitor):
    launch_args = ["--start-maximized"]
    if config["use_adblock"]:
        print("BROWSER: Edge — Ad blocker (uBlock Origin) ENABLED.")
        launch_args.extend([
            f"--disable-extensions-except={UBLOCK_EDGE_PATH}",
            f"--load-extension={UBLOCK_EDGE_PATH}"
        ])
    else:
        print("BROWSER: Edge — Ad blocker DISABLED.")

    context = p.chromium.launch_persistent_context(
        USER_DATA_DIR_EDGE,
        headless=False,
        channel="msedge",          # Playwright's channel name for Edge
        args=launch_args,
        no_viewport=True
    )
    _launch_and_run(context, config, network_monitor)

def _run_firefox_context(p, config, network_monitor):
    profile = USER_DATA_DIR_FIREFOX_ADBLOCK if config["use_adblock"] else USER_DATA_DIR_FIREFOX_NOADBLOCK
    status = "ENABLED" if config["use_adblock"] else "DISABLED"
    print(f"BROWSER: Firefox — Ad blocker {status}.")

    context = p.firefox.launch_persistent_context(
        profile,
        headless=False,
        no_viewport=True
    )
    _launch_and_run(context, config, network_monitor)

def reset_firefox_profiles():
    """Resets working Firefox profiles from the master copies."""
    for src, dst in [
        (FIREFOX_ADBLOCK_MASTER_PROFILE, USER_DATA_DIR_FIREFOX_ADBLOCK),
        (FIREFOX_NOADBLOCK_MASTER_PROFILE, USER_DATA_DIR_FIREFOX_NOADBLOCK),
    ]:
        if not os.path.exists(src):
            print(f"[ERROR] Firefox master profile not found at '{src}'. Did you run the setup step?")
            sys.exit(1)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    print("Firefox profiles reset from master copies.")

# Main 
def main():
    """Main function to orchestrate the experiments."""
    if not ctypes.windll.shell32.IsUserAnAdmin():
        print("[ERROR] This script requires Administrator privileges to access hardware sensors.")
        return

    # Set trial run or not, testing a single website for a single sample
    TRIAL_RUN = False

    if TRIAL_RUN:
        URLs = SITES_TO_TEST
        N = 3
    else:
        URLs = SITES_TO_TEST
        N = 20

    # Generate the list of experiments
    base_experiments = []
    for site in URLs:
        site_name = site['name']
        actions = site['actions']
        for url in site['urls']:
            for browser in ["chrome", "edge", "firefox"]:
                for use_adblock in [False, True]:
                    label = "with_adblock" if use_adblock else "no_adblock"
                    base_experiments.append({
                        "name": f"{site_name}_{browser}_{label}",
                        "url": url,
                        "duration_seconds": DURATION_SECONDS,
                        "use_adblock": use_adblock,
                        "browser": browser,
                        "output_file": OUTPUT_FILE,
                        "actions": actions
                    })

    EXPERIMENTS = []
    run_id_counter = 0
    for i in range(N): # Loop N times for N trials
        for base_exp in base_experiments:
            # Create a copy 
            new_exp_trial = base_exp.copy()
            # Assign the unique run_id to the new copy
            new_exp_trial['run_id'] = run_id_counter
            EXPERIMENTS.append(new_exp_trial)
            run_id_counter += 1

    total_duration_hrs = len(EXPERIMENTS * (DURATION_SECONDS + COOLDOWN_SECONDS)) / 3600

    if input(f"Estimated duration of experiment is AT LEAST {math.floor(total_duration_hrs):.2f} hours and {(total_duration_hrs % 1) * 60:.2f} minutes. Continue? [y/n] ").lower() != "y":
        sys.exit()


    # Shuffle!
    random.shuffle(EXPERIMENTS)

    # Delete old folders
    if os.path.exists(USER_DATA_DIR_CHROME):
        shutil.rmtree(USER_DATA_DIR_CHROME)
    if os.path.exists(USER_DATA_DIR_EDGE):
        shutil.rmtree(USER_DATA_DIR_EDGE)
    reset_firefox_profiles()
        
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    hw_monitor = HardwareMonitor(DLL_PATH)
    hw_monitor.initialize_and_discover()

    if WARMUP_SECONDS > 0 and not TRIAL_RUN:
        warmup_cpu(duration_seconds=WARMUP_SECONDS)
        warmup_gpu(duration_seconds=WARMUP_SECONDS)

    if not EXPERIMENTS:
        print("[ERROR] The EXPERIMENTS list is empty. Check your SITES_TO_TEST configuration and script logic.")
        return

    for i, experiment_config in enumerate(EXPERIMENTS):
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT {i+1}/{len(EXPERIMENTS)}: {experiment_config['name']}")
        print("="*80)
        
        stop_monitoring_event = threading.Event()
        
        # Create a network monitor instance for each experiment
        network_monitor = NetworkTrafficMonitor(AD_BLOCKLIST_FILE)
        
        # Pass the shared network_monitor instance to both threads
        monitor_thread = threading.Thread(target=run_monitoring_task, args=(hw_monitor, network_monitor, experiment_config, stop_monitoring_event))
        browser_thread = threading.Thread(target=run_browser_task, args=(experiment_config, network_monitor, stop_monitoring_event))
        
        monitor_thread.start()
        time.sleep(2)
        browser_thread.start()
        
        browser_thread.join()
        monitor_thread.join()
        
        # Log final network stats for a summary view
        final_stats = network_monitor.get_current_stats()
        print("--- Network Summary for this Experiment ---")
        print(f"Total Requests: {final_stats['total_requests']}")
        print(f"Total MB Downloaded: {final_stats['total_bytes'] / (1024*1024):.2f} MB")
        print(f"Ad-related Requests (from list): {final_stats['ad_requests']}")
        print(f"Ad-related MB Downloaded (from list): {final_stats['ad_bytes'] / (1024*1024):.2f} MB")
        print(f"Requests Blocked by Extension: {final_stats['blocked_by_client_requests']}")
        print("-----------------------------------------")
        
        print(f"Experiment '{experiment_config['name']}' complete. ---")

        if i < len(EXPERIMENTS) - 1:
            print(f"\nSystem cooling down for {COOLDOWN_SECONDS} seconds before next experiment...")
            time.sleep(COOLDOWN_SECONDS)

    hw_monitor.close()
    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()