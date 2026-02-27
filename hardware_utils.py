# hardware_utils.py

import clr
import sys
import time
import re
import multiprocessing
import numba
from numba import cuda
import psutil
import os

def worker(target_load, stop_event):
    """
    A function designed to run in a separate process to generate CPU load.
    It will try to maintain a CPU load above a certain threshold.
    """
    # On Windows, spawned processes might inherit console handles,
    # which can be problematic. This is a good practice, though not
    # strictly necessary to fix the pickling error.
    os.setsid() if hasattr(os, 'setsid') else None

    while not stop_event.is_set():
        # Check current CPU utilization of the entire system.
        # Note: psutil.cpu_percent() measures system-wide CPU usage.
        # A single process aiming for 80% load on a 16-core machine
        # will struggle to make the system-wide average hit 80%.
        # The logic here is a simple "burn if below target" approach.
        if psutil.cpu_percent(interval=0.1) < target_load * 100:
            # Intense calculation to generate load
            start_time = time.time()
            while (time.time() - start_time) < 0.1: # Burn for a short burst
                _ = [x*x for x in range(1000)]
        else:
            # If system load is already high, sleep to avoid overshooting
            time.sleep(0.1)

def warmup_cpu(duration_seconds: int, target_load: float = 0.8, num_cores: int = None):
    """
    Warms up the CPU by creating processes to generate load.

    Args:
        duration_seconds: The number of seconds to run the warm-up.
        target_load: The target CPU load as a fraction (e.g., 0.8 for 80%).
        num_cores: The number of CPU cores to target. Defaults to all available cores.
    """
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    print(f"--- Starting CPU Warm-up ---")
    print(f"Targeting {num_cores} cores for {duration_seconds} seconds at {target_load*100:.0f}% load.")

    processes = []
    stop_event = multiprocessing.Event()

    for _ in range(num_cores):
        # The 'target' is now the top-level 'worker' function.
        # We pass its arguments via the 'args' tuple.
        process = multiprocessing.Process(target=worker, args=(target_load, stop_event))
        processes.append(process)
        process.start() # This will now work correctly

    try:
        # Let the worker processes run for the specified duration
        time.sleep(duration_seconds)
    finally:
        # Cleanly stop all worker processes
        print("--- Stopping CPU Warm-up ---")
        stop_event.set()
        for process in processes:
            process.join(timeout=5) # Wait for each process to finish
            if process.is_alive():
                print(f"Process {process.pid} did not terminate gracefully, terminating now.")
                process.terminate() # Forcefully terminate if it doesn't join

    print("--- CPU Warm-up Complete ---")


def warmup_gpu(duration_seconds=30):
    """
    Applies a consistent load to the NVIDIA GPU using Numba and CUDA.
    If no compatible GPU is found, it will print a message and return.
    """
    print(f"\n--- Starting GPU Warm-up ---")
    if not cuda.is_available():
        print("NVIDIA CUDA GPU not available or Numba cannot detect it. Skipping GPU warm-up.")
        return

    print(f"Found compatible CUDA GPU. Warming up for {duration_seconds} seconds.")

    @cuda.jit
    def gpu_busy_kernel(data):
        """A simple CUDA kernel to perform some math operations."""
        idx = cuda.grid(1)
        if idx < data.size:
            data[idx] = numba.float32(numba.cos(data[idx])**2 + numba.sin(data[idx])**2)

    # Use a reasonably large array to ensure the GPU is kept busy
    data_size = 1024 * 1024 * 32 # 32M elements
    data_array = numba.float32([i for i in range(data_size)])
    
    # Move data to the GPU device
    device_array = cuda.to_device(data_array)

    threads_per_block = 256
    blocks_per_grid = (device_array.size + (threads_per_block - 1)) // threads_per_block

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        # Repeatedly launch the kernel to keep the GPU busy
        gpu_busy_kernel[blocks_per_grid, threads_per_block](device_array)
        # It's crucial to synchronize to ensure the kernel call completes
        # before the next Python loop iteration. This keeps the GPU fed.
        cuda.synchronize()
    
    print("--- GPU Warm-up Complete ---")


class HardwareMonitor:
    """A class to initialize LHM, discover all sensors, and collect metrics."""
    def __init__(self, dll_path):
        self.dll_path = dll_path
        self.computer = None
        self.sensors = [] # This will now be a list of sensor dicts
        self.sensor_headers = [] # This will store the unique IDs for CSV headers
        
        try:
            clr.AddReference(self.dll_path)
            from LibreHardwareMonitor import Hardware
            self.Hardware = Hardware
        except Exception as e:
            print(f"Error initializing LibreHardwareMonitor: {e}")
            sys.exit(1)

    def initialize_and_discover(self):
        """Calls the utility to discover all sensors."""
        print("MONITOR: Starting automatic sensor discovery...")
        self.sensors = discover_sensors(self.dll_path)
        if not self.sensors:
            print("MONITOR: CRITICAL - No sensors were discovered. Exiting.")
            sys.exit(1)
        
        self.sensor_headers = [s['id'] for s in self.sensors]
        print(f"MONITOR: Discovery complete. Found {len(self.sensors)} sensors to monitor.")
        
        # We need to create and open a persistent computer object for polling
        self.computer = self.Hardware.Computer()
        self.computer.IsCpuEnabled = True
        self.computer.IsGpuEnabled = True
        self.computer.IsMemoryEnabled = True
        self.computer.IsMotherboardEnabled = True
        self.computer.IsStorageEnabled = True
        self.computer.Open()
        time.sleep(2) # Give it a moment to stabilize

    def get_all_metrics(self):
        """Retrieves current values from ALL discovered sensors."""
        if not self.computer:
            return {}
            
        # We must update the main computer object, which cascades to all hardware
        for hw in self.computer.Hardware:
            hw.Update()
            for sub_hw in hw.SubHardware:
                sub_hw.Update()

        metrics = {}
        for sensor_info in self.sensors:
            sensor_obj = sensor_info['object']
            # The sensor objects we have are from a closed 'computer' instance.
            # We need to find the *corresponding live sensor* in our open instance.
            # This is a bit complex but necessary.
            live_sensor = self._find_live_sensor(sensor_info)
            if live_sensor and live_sensor.Value is not None:
                metrics[sensor_info['id']] = live_sensor.Value
            else:
                metrics[sensor_info['id']] = 0.0
        return metrics

    def _find_live_sensor(self, sensor_info_to_find):
        """Finds the 'live' sensor object corresponding to a discovered one."""
        for hw in self.computer.Hardware:
            # Check direct sensors
            for s in hw.Sensors:
                if s.Identifier == sensor_info_to_find['object'].Identifier:
                    return s
            # Check sub-hardware sensors
            for sub_hw in hw.SubHardware:
                for s in sub_hw.Sensors:
                    if s.Identifier == sensor_info_to_find['object'].Identifier:
                        return s
        return None

    def close(self):
        """Closes the handle to LibreHardwareMonitor."""
        if self.computer:
            self.computer.Close()
            print("MONITOR: Hardware monitor handle closed.")



def discover_sensors(dll_path):
    """
    Connects to LibreHardwareMonitor, scans all hardware, and returns a
    structured list of all found sensors.

    Args:
        dll_path (str): The absolute path to the LibreHardwareMonitorLib.dll.

    Returns:
        list: A list of dictionaries, where each dictionary represents a sensor
              and contains its unique ID, the sensor object, and its type.
              Returns an empty list if initialization fails.
    """
    try:
        clr.AddReference(dll_path)
        from LibreHardwareMonitor import Hardware
    except Exception as e:
        print(f"[Hardware Utils] Error adding DLL reference: {e}")
        return []

    computer = Hardware.Computer()
    computer.IsCpuEnabled = True
    computer.IsGpuEnabled = True
    computer.IsMemoryEnabled = True
    computer.IsMotherboardEnabled = True # Enable for more sensors
    computer.IsStorageEnabled = True    # Enable for more sensors
    
    print("[Hardware Utils] Opening LibreHardwareMonitor handle for sensor discovery...")
    computer.Open()
    time.sleep(2) # Allow time for initialization

    discovered_sensors = []

    for hw in computer.Hardware:
        hw.Update()
        process_hardware(hw, discovered_sensors)

    computer.Close()
    print(f"[Hardware Utils] Sensor discovery complete. Found {len(discovered_sensors)} sensors.")
    return discovered_sensors

def process_hardware(hw, sensor_list):
    """
    Recursively processes a hardware item and its sub-hardware to find sensors.
    """
    # Process sensors on the current hardware item
    for sensor in hw.Sensors:
        # Create a unique, file-system-friendly ID for each sensor
        # e.g., "CPU_Intel_Core_i9_9900K_CPU_Total_Load"
        hw_name_safe = re.sub(r'[^a-zA-Z0-9_]', '_', hw.Name)
        sensor_name_safe = re.sub(r'[^a-zA-Z0-9_]', '_', sensor.Name)
        
        unique_id = f"{hw.HardwareType}_{hw_name_safe}_{sensor_name_safe}_{sensor.SensorType}"
        
        sensor_list.append({
            "id": unique_id,
            "object": sensor,
            "type": sensor.SensorType.ToString(),
            "hw_name": hw.Name
        })

    # Recursively process sub-hardware
    for sub_hw in hw.SubHardware:
        sub_hw.Update()
        process_hardware(sub_hw, sensor_list)