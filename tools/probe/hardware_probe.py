"""Hardware and Display/GPU Routing Probe for OpenPrism Research System.

Records CPU/GPU/display topology, refresh rates, dGPU/MUX routing,
CUDA identity, power mode, and evaluates whether zero-copy capture is possible.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import dataclasses
from datetime import datetime, timezone
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


# --- Windows Win32 / DXGI / CUDA ctypes Structures and Constants ---

class LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
    ]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class DXGI_OUTPUT_DESC(ctypes.Structure):
    _fields_ = [
        ("DeviceName", ctypes.c_wchar * 32),
        ("DesktopCoordinates", RECT),
        ("AttachedToDesktop", wintypes.BOOL),
        ("Rotation", wintypes.UINT),
        ("Monitor", wintypes.HANDLE),
    ]


class DEVMODEW(ctypes.Structure):
    _fields_ = [
        ("dmDeviceName", ctypes.c_wchar * 32),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmOrientation", ctypes.c_short),
        ("dmPaperSize", ctypes.c_short),
        ("dmPaperLength", ctypes.c_short),
        ("dmPaperWidth", ctypes.c_short),
        ("dmScale", ctypes.c_short),
        ("dmCopies", ctypes.c_short),
        ("dmDefaultSource", ctypes.c_short),
        ("dmPrintQuality", ctypes.c_short),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", ctypes.c_wchar * 32),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    ]


class DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
    ]


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE),
        ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE),
        ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


# COM GUID for IDXGIFactory2: {50c83a1c-e072-4c48-87b0-3630fa36a6d0}
IID_IDXGIFactory2 = GUID(
    0x50C83A1C, 0xE072, 0x4C48, (ctypes.c_ubyte * 8)(0x87, 0xB0, 0x36, 0x30, 0xFA, 0x36, 0xA6, 0xD0)
)
# COM GUID for IDXGIFactory1: {770aae78-f26f-4ada-aaa0-41e56837b316}
IID_IDXGIFactory1 = GUID(
    0x770AAE78, 0xF26F, 0x4ADA, (ctypes.c_ubyte * 8)(0xAA, 0xA0, 0x41, 0xE5, 0x68, 0x37, 0xB3, 0x16)
)


def format_luid(low: int, high: int) -> str:
    """Format LUID as a normalized hex string."""
    return f"0x{high:08x}:{low:08x}"


# --- Dataclasses for Environment Manifest ---

@dataclasses.dataclass
class SystemInfo:
    os_caption: str
    os_version: str
    os_build: str
    architecture: str
    hostname: str
    probe_timestamp_utc: str


@dataclasses.dataclass
class CpuInfo:
    name: str
    cores: int
    logical_processors: int
    max_clock_mhz: int
    architecture: str


@dataclasses.dataclass
class GpuAdapterInfo:
    index: int
    description: str
    vendor_id_hex: str
    device_id_hex: str
    sub_sys_id_hex: str
    revision: int
    dedicated_video_memory_mb: int
    dedicated_system_memory_mb: int
    shared_system_memory_mb: int
    luid_low: int
    luid_high: int
    luid: str
    is_software: bool


@dataclasses.dataclass
class DisplayOutputInfo:
    device_name: str
    monitor_name: str
    is_attached: bool
    is_primary: bool
    width_px: int
    height_px: int
    refresh_rate_hz: int
    bits_per_pixel: int
    desktop_bounds: Dict[str, int]
    adapter_luid: str
    adapter_description: str


@dataclasses.dataclass
class CudaDeviceInfo:
    device_index: int
    name: str
    compute_capability_major: int
    compute_capability_minor: int
    multiprocessor_count: int
    clock_rate_mhz: float
    total_memory_mb: int
    luid_low: int
    luid_high: int
    luid: str
    pci_bus_id: str


@dataclasses.dataclass
class PowerInfo:
    ac_line_status: str  # "Online" | "Battery" | "Unknown"
    battery_percent: int
    power_scheme_guid: str
    power_scheme_name: str


@dataclasses.dataclass
class TopologyEvaluation:
    primary_display_device: Optional[str]
    primary_display_adapter_luid: Optional[str]
    primary_display_adapter_name: Optional[str]
    primary_cuda_device_index: Optional[int]
    primary_cuda_device_name: Optional[str]
    primary_cuda_adapter_luid: Optional[str]
    luid_match: bool
    topology_type: str  # "SAME_ADAPTER_DIRECT" | "CROSS_ADAPTER_HYBRID" | "NO_CUDA_DEVICE" | "NO_ATTACHED_DISPLAYS"
    zero_copy_capture_possible: bool
    requires_cross_adapter_fallback: bool
    diagnosis: str
    recommendations: List[str]


@dataclasses.dataclass
class EnvironmentManifest:
    schema_version: int
    system: SystemInfo
    cpu: CpuInfo
    gpus: List[GpuAdapterInfo]
    displays: List[DisplayOutputInfo]
    cuda_devices: List[CudaDeviceInfo]
    power: PowerInfo
    topology: TopologyEvaluation

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --- Pure Analytical Functions (Hardware-Independent & Unit-Testable) ---

def evaluate_topology(
    displays: List[DisplayOutputInfo],
    gpus: List[GpuAdapterInfo],
    cuda_devices: List[CudaDeviceInfo],
) -> TopologyEvaluation:
    """Evaluate display adapter and CUDA device topology to determine zero-copy capture feasibility.

    Pure function with no OS side-effects, fully unit-testable.
    """
    attached_displays = [d for d in displays if d.is_attached]
    if not attached_displays:
        return TopologyEvaluation(
            primary_display_device=None,
            primary_display_adapter_luid=None,
            primary_display_adapter_name=None,
            primary_cuda_device_index=cuda_devices[0].device_index if cuda_devices else None,
            primary_cuda_device_name=cuda_devices[0].name if cuda_devices else None,
            primary_cuda_adapter_luid=cuda_devices[0].luid if cuda_devices else None,
            luid_match=False,
            topology_type="NO_ATTACHED_DISPLAYS",
            zero_copy_capture_possible=False,
            requires_cross_adapter_fallback=True,
            diagnosis="No active attached display outputs found.",
            recommendations=["Attach or enable a display output before running capture pipelines."],
        )

    # Find primary display (or first attached display)
    primary_display = next((d for d in attached_displays if d.is_primary), attached_displays[0])

    if not cuda_devices:
        return TopologyEvaluation(
            primary_display_device=primary_display.device_name,
            primary_display_adapter_luid=primary_display.adapter_luid,
            primary_display_adapter_name=primary_display.adapter_description,
            primary_cuda_device_index=None,
            primary_cuda_device_name=None,
            primary_cuda_adapter_luid=None,
            luid_match=False,
            topology_type="NO_CUDA_DEVICE",
            zero_copy_capture_possible=False,
            requires_cross_adapter_fallback=False,
            diagnosis="No NVIDIA CUDA device detected on the system.",
            recommendations=[
                "Verify NVIDIA GPU drivers and CUDA installation.",
                "Ensure dedicated GPU is enabled in device manager.",
            ],
        )

    primary_cuda = cuda_devices[0]

    # Compare LUIDs
    luid_match = (
        primary_display.adapter_luid.lower() == primary_cuda.luid.lower()
        and primary_cuda.luid.lower() not in ("0x00000000:00000000", "")
    )

    if luid_match:
        return TopologyEvaluation(
            primary_display_device=primary_display.device_name,
            primary_display_adapter_luid=primary_display.adapter_luid,
            primary_display_adapter_name=primary_display.adapter_description,
            primary_cuda_device_index=primary_cuda.device_index,
            primary_cuda_device_name=primary_cuda.name,
            primary_cuda_adapter_luid=primary_cuda.luid,
            luid_match=True,
            topology_type="SAME_ADAPTER_DIRECT",
            zero_copy_capture_possible=True,
            requires_cross_adapter_fallback=False,
            diagnosis=(
                f"Display '{primary_display.device_name}' and CUDA Device {primary_cuda.device_index} "
                f"('{primary_cuda.name}') share the same adapter LUID ({primary_cuda.luid}). "
                f"Direct D3D11-CUDA interop zero-copy capture is supported."
            ),
            recommendations=[
                "Use direct DXGI Desktop Duplication with D3D11-CUDA interop for lowest capture latency.",
                "Allocate D3D11 capture textures directly on the dedicated GPU.",
            ],
        )
    else:
        return TopologyEvaluation(
            primary_display_device=primary_display.device_name,
            primary_display_adapter_luid=primary_display.adapter_luid,
            primary_display_adapter_name=primary_display.adapter_description,
            primary_cuda_device_index=primary_cuda.device_index,
            primary_cuda_device_name=primary_cuda.name,
            primary_cuda_adapter_luid=primary_cuda.luid,
            luid_match=False,
            topology_type="CROSS_ADAPTER_HYBRID",
            zero_copy_capture_possible=False,
            requires_cross_adapter_fallback=True,
            diagnosis=(
                f"Hybrid GPU laptop topology detected: Display '{primary_display.device_name}' is driven by "
                f"'{primary_display.adapter_description}' (LUID {primary_display.adapter_luid}), whereas CUDA runs on "
                f"'{primary_cuda.name}' (LUID {primary_cuda.luid}). Direct same-adapter D3D11-CUDA interop cannot "
                f"bind across different hardware adapters without a cross-adapter transfer."
            ),
            recommendations=[
                "Option 1 (Direct zero-copy): Enable dGPU MUX switch / Advanced Optimus in BIOS or OEM software to route internal display directly to NVIDIA dGPU.",
                "Option 2 (Direct zero-copy): Connect an external high-refresh monitor to the physical port wired directly to the NVIDIA dGPU.",
                "Option 3 (Hybrid fallback): Use Windows Graphics Capture (WGC) or cross-adapter staging texture transfer with measured latency monitoring (M0-03/M2).",
            ],
        )


# --- Platform Probing Functions ---

def probe_system_info() -> SystemInfo:
    """Collect OS and host information."""
    os_caption = f"{platform.system()} {platform.release()} ({platform.version()})"
    os_build = platform.version()

    # Try querying Win32_OperatingSystem via PowerShell / CIM
    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object -Property Caption, Version, BuildNumber | ConvertTo-Json"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            os_caption = data.get("Caption", os_caption).strip()
            os_build = str(data.get("BuildNumber", os_build)).strip()
    except Exception:
        pass

    return SystemInfo(
        os_caption=os_caption,
        os_version=platform.version(),
        os_build=os_build,
        architecture=platform.machine(),
        hostname=platform.node(),
        probe_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def probe_cpu_info() -> CpuInfo:
    """Collect CPU topology and capabilities."""
    name = platform.processor() or "Unknown CPU"
    cores = os.cpu_count() or 1
    logical = cores
    max_clock = 0

    try:
        cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | ConvertTo-Json"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout.strip():
            raw = json.loads(proc.stdout)
            # When multiple CPU sockets exist, CIM returns a list
            data = raw[0] if isinstance(raw, list) else raw
            name = data.get("Name", name).strip()
            cores = int(data.get("NumberOfCores", cores))
            logical = int(data.get("NumberOfLogicalProcessors", logical))
            max_clock = int(data.get("MaxClockSpeed", 0))
    except Exception:
        pass

    return CpuInfo(
        name=name,
        cores=cores,
        logical_processors=logical,
        max_clock_mhz=max_clock,
        architecture=platform.machine(),
    )


def probe_power_info() -> PowerInfo:
    """Collect AC power status and active Windows power scheme."""
    ac_line = "Unknown"
    battery_pct = 255

    try:
        sps = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
            if sps.ACLineStatus == 1:
                ac_line = "Online"
            elif sps.ACLineStatus == 0:
                ac_line = "Battery"
            battery_pct = int(sps.BatteryLifePercent)
    except Exception:
        pass

    scheme_guid = ""
    scheme_name = ""
    try:
        proc = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            match = re.search(r"GUID:\s*([0-9a-fA-F-]+)\s*\((.*?)\)", proc.stdout)
            if match:
                scheme_guid = match.group(1).strip()
                scheme_name = match.group(2).strip()
    except Exception:
        pass

    return PowerInfo(
        ac_line_status=ac_line,
        battery_percent=battery_pct,
        power_scheme_guid=scheme_guid,
        power_scheme_name=scheme_name,
    )


def probe_dxgi_and_displays() -> Tuple[List[GpuAdapterInfo], List[DisplayOutputInfo]]:
    """Enumerate all DXGI adapters and display outputs."""
    gpus: List[GpuAdapterInfo] = []
    displays: List[DisplayOutputInfo] = []

    # Map display device names to modes via EnumDisplaySettings
    display_modes: Dict[str, Dict[str, Any]] = {}
    display_monitors: Dict[str, str] = {}

    try:
        user32 = ctypes.windll.user32
        dev_idx = 0
        while True:
            dd = DISPLAY_DEVICEW()
            dd.cb = ctypes.sizeof(DISPLAY_DEVICEW)
            if not user32.EnumDisplayDevicesW(None, dev_idx, ctypes.byref(dd), 0):
                break

            dname = dd.DeviceName
            dm = DEVMODEW()
            dm.dmSize = ctypes.sizeof(DEVMODEW)
            if user32.EnumDisplaySettingsW(dname, -1, ctypes.byref(dm)):
                display_modes[dname] = {
                    "width": dm.dmPelsWidth,
                    "height": dm.dmPelsHeight,
                    "refresh_rate": dm.dmDisplayFrequency,
                    "bpp": dm.dmBitsPerPel,
                    "is_primary": bool(dd.StateFlags & 0x00000004),
                    "is_attached": bool(dd.StateFlags & 0x00000001),
                }

            mon_idx = 0
            while True:
                mon = DISPLAY_DEVICEW()
                mon.cb = ctypes.sizeof(DISPLAY_DEVICEW)
                if not user32.EnumDisplayDevicesW(dname, mon_idx, ctypes.byref(mon), 0):
                    break
                if mon.DeviceString:
                    display_monitors[dname] = mon.DeviceString
                mon_idx += 1

            dev_idx += 1
    except Exception:
        pass

    # Query DXGI adapters and outputs via DXGI COM interface
    try:
        dxgi = ctypes.windll.dxgi
        factory_p = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory2), ctypes.byref(factory_p))
        if hr != 0:
            hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory_p))

        if hr == 0 and factory_p.value:
            factory_vtbl = ctypes.cast(
                ctypes.cast(factory_p, ctypes.POINTER(ctypes.c_void_p)).contents,
                ctypes.POINTER(ctypes.c_void_p),
            )
            # IDXGIFactory1::EnumAdapters1 is at index 12
            EnumAdapters1_proto = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)
            )
            EnumAdapters1 = EnumAdapters1_proto(factory_vtbl[12])
            Release_proto = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)
            ReleaseFactory = Release_proto(factory_vtbl[2])

            adapter_idx = 0
            while True:
                adapter_p = ctypes.c_void_p()
                hr_adapt = EnumAdapters1(factory_p, adapter_idx, ctypes.byref(adapter_p))
                if hr_adapt != 0 or not adapter_p.value:
                    break

                adapt_vtbl = ctypes.cast(
                    ctypes.cast(adapter_p, ctypes.POINTER(ctypes.c_void_p)).contents,
                    ctypes.POINTER(ctypes.c_void_p),
                )
                GetDesc1_proto = ctypes.WINFUNCTYPE(
                    ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1)
                )
                GetDesc1 = GetDesc1_proto(adapt_vtbl[10])
                EnumOutputs_proto = ctypes.WINFUNCTYPE(
                    ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)
                )
                EnumOutputs = EnumOutputs_proto(adapt_vtbl[7])
                ReleaseAdapter = Release_proto(adapt_vtbl[2])

                desc = DXGI_ADAPTER_DESC1()
                GetDesc1(adapter_p, ctypes.byref(desc))

                luid_str = format_luid(desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart)
                is_sw = bool(desc.Flags & 0x2)  # DXGI_ADAPTER_FLAG_SOFTWARE

                gpu_info = GpuAdapterInfo(
                    index=adapter_idx,
                    description=desc.Description.strip(),
                    vendor_id_hex=f"0x{desc.VendorId:04x}",
                    device_id_hex=f"0x{desc.DeviceId:04x}",
                    sub_sys_id_hex=f"0x{desc.SubSysId:08x}",
                    revision=desc.Revision,
                    dedicated_video_memory_mb=int(desc.DedicatedVideoMemory // (1024 * 1024)),
                    dedicated_system_memory_mb=int(desc.DedicatedSystemMemory // (1024 * 1024)),
                    shared_system_memory_mb=int(desc.SharedSystemMemory // (1024 * 1024)),
                    luid_low=desc.AdapterLuid.LowPart,
                    luid_high=desc.AdapterLuid.HighPart,
                    luid=luid_str,
                    is_software=is_sw,
                )
                gpus.append(gpu_info)

                out_idx = 0
                while True:
                    out_p = ctypes.c_void_p()
                    hr_out = EnumOutputs(adapter_p, out_idx, ctypes.byref(out_p))
                    if hr_out != 0 or not out_p.value:
                        break

                    out_vtbl = ctypes.cast(
                        ctypes.cast(out_p, ctypes.POINTER(ctypes.c_void_p)).contents,
                        ctypes.POINTER(ctypes.c_void_p),
                    )
                    GetDescOut_proto = ctypes.WINFUNCTYPE(
                        ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_OUTPUT_DESC)
                    )
                    GetDescOut = GetDescOut_proto(out_vtbl[7])
                    ReleaseOutput = Release_proto(out_vtbl[2])

                    odesc = DXGI_OUTPUT_DESC()
                    GetDescOut(out_p, ctypes.byref(odesc))

                    dev_name = odesc.DeviceName.strip()
                    mode = display_modes.get(dev_name, {})
                    mon_name = display_monitors.get(dev_name, "Generic PnP Monitor")

                    # Coordinates and dimensions
                    left = odesc.DesktopCoordinates.left
                    top = odesc.DesktopCoordinates.top
                    right = odesc.DesktopCoordinates.right
                    bottom = odesc.DesktopCoordinates.bottom

                    width = mode.get("width", right - left)
                    height = mode.get("height", bottom - top)
                    refresh = mode.get("refresh_rate", 60)
                    bpp = mode.get("bpp", 32)
                    is_primary = mode.get("is_primary", (left == 0 and top == 0))
                    is_attached = bool(odesc.AttachedToDesktop)

                    disp_info = DisplayOutputInfo(
                        device_name=dev_name,
                        monitor_name=mon_name,
                        is_attached=is_attached,
                        is_primary=is_primary,
                        width_px=width,
                        height_px=height,
                        refresh_rate_hz=refresh,
                        bits_per_pixel=bpp,
                        desktop_bounds={"left": left, "top": top, "right": right, "bottom": bottom},
                        adapter_luid=luid_str,
                        adapter_description=desc.Description.strip(),
                    )
                    displays.append(disp_info)

                    ReleaseOutput(out_p)
                    out_idx += 1

                ReleaseAdapter(adapter_p)
                adapter_idx += 1

            ReleaseFactory(factory_p)
    except Exception as ex:
        print(f"Warning: DXGI enumeration failed: {ex}", file=sys.stderr)

    return gpus, displays


def probe_cuda_devices() -> List[CudaDeviceInfo]:
    """Enumerate CUDA devices via CUDA driver API (nvcuda.dll)."""
    cuda_devices: List[CudaDeviceInfo] = []

    try:
        nvcuda = ctypes.windll.nvcuda
        cuInit = nvcuda.cuInit
        cuInit.restype = ctypes.c_int
        if cuInit(0) != 0:
            return cuda_devices

        count = ctypes.c_int()
        if nvcuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value <= 0:
            return cuda_devices

        # Query nvidia-smi for PCI bus IDs if available
        pci_bus_map: Dict[str, str] = {}
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        pci_bus_map[parts[0]] = parts[1]
        except Exception:
            pass

        for i in range(count.value):
            dev = ctypes.c_int()
            if nvcuda.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue

            name_buf = ctypes.create_string_buffer(256)
            nvcuda.cuDeviceGetName(name_buf, 256, dev)
            name_str = name_buf.value.decode("utf-8", errors="replace").strip()

            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
            # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
            nvcuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)
            nvcuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)

            # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            sm_count = ctypes.c_int(0)
            nvcuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, dev)

            # CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13 (kHz)
            clock_rate = ctypes.c_int(0)
            nvcuda.cuDeviceGetAttribute(ctypes.byref(clock_rate), 13, dev)

            # Total memory
            total_bytes = ctypes.c_uint64(0)
            if hasattr(nvcuda, "cuDeviceTotalMem_v2"):
                nvcuda.cuDeviceTotalMem_v2(ctypes.byref(total_bytes), dev)
            else:
                nvcuda.cuDeviceTotalMem(ctypes.byref(total_bytes), dev)

            # LUID
            luid_low = 0
            luid_high = 0
            luid_str = "0x00000000:00000000"
            if hasattr(nvcuda, "cuDeviceGetLuid"):
                luid_bytes = (ctypes.c_char * 8)()
                node_mask = ctypes.c_uint()
                if nvcuda.cuDeviceGetLuid(luid_bytes, ctypes.byref(node_mask), dev) == 0:
                    import struct
                    luid_low, luid_high = struct.unpack("<II", luid_bytes.raw)
                    luid_str = format_luid(luid_low, luid_high)

            pci_id = pci_bus_map.get(str(i), "Unknown")

            cuda_info = CudaDeviceInfo(
                device_index=i,
                name=name_str,
                compute_capability_major=major.value,
                compute_capability_minor=minor.value,
                multiprocessor_count=sm_count.value,
                clock_rate_mhz=round(clock_rate.value / 1000.0, 1),
                total_memory_mb=int(total_bytes.value // (1024 * 1024)),
                luid_low=luid_low,
                luid_high=luid_high,
                luid=luid_str,
                pci_bus_id=pci_id,
            )
            cuda_devices.append(cuda_info)
    except Exception as ex:
        print(f"Warning: CUDA driver probe failed: {ex}", file=sys.stderr)

    return cuda_devices


def collect_environment_manifest() -> EnvironmentManifest:
    """Collect comprehensive hardware, OS, GPU, display, and CUDA manifest."""
    sys_info = probe_system_info()
    cpu_info = probe_cpu_info()
    gpus, displays = probe_dxgi_and_displays()
    cuda_devices = probe_cuda_devices()
    power_info = probe_power_info()
    topology = evaluate_topology(displays, gpus, cuda_devices)

    return EnvironmentManifest(
        schema_version=1,
        system=sys_info,
        cpu=cpu_info,
        gpus=gpus,
        displays=displays,
        cuda_devices=cuda_devices,
        power=power_info,
        topology=topology,
    )


def print_manifest_summary(manifest: EnvironmentManifest) -> None:
    """Format and print a readable summary report of the environment."""
    top = manifest.topology
    print("================================================================================")
    print(" OPENPRISM HARDWARE AND DISPLAY/GPU ROUTING PROBE")
    print("================================================================================")
    print(f"Timestamp:   {manifest.system.probe_timestamp_utc}")
    print(f"Host:        {manifest.system.hostname} ({manifest.system.os_caption})")
    print(f"CPU:         {manifest.cpu.name} ({manifest.cpu.cores} cores / {manifest.cpu.logical_processors} threads)")
    print(f"Power:       AC Status: {manifest.power.ac_line_status} | Battery: {manifest.power.battery_percent}% | Scheme: {manifest.power.power_scheme_name}")
    print("--------------------------------------------------------------------------------")
    print(" DXGI GPU ADAPTERS:")
    for gpu in manifest.gpus:
        sw_tag = " [SOFTWARE]" if gpu.is_software else ""
        print(f"  [{gpu.index}] {gpu.description}{sw_tag}")
        print(f"      Vendor: {gpu.vendor_id_hex} | Device: {gpu.device_id_hex} | LUID: {gpu.luid}")
        print(f"      Dedicated VRAM: {gpu.dedicated_video_memory_mb} MB | Shared RAM: {gpu.shared_system_memory_mb} MB")

    print("--------------------------------------------------------------------------------")
    print(" DISPLAY OUTPUTS:")
    for disp in manifest.displays:
        prim_tag = " (Primary)" if disp.is_primary else ""
        attach_tag = "Attached" if disp.is_attached else "Detached"
        print(f"  - {disp.device_name}: {disp.monitor_name}{prim_tag} [{attach_tag}]")
        print(f"      Mode: {disp.width_px}x{disp.height_px} @ {disp.refresh_rate_hz} Hz ({disp.bits_per_pixel} bpp)")
        print(f"      Driving Adapter: {disp.adapter_description} (LUID: {disp.adapter_luid})")

    print("--------------------------------------------------------------------------------")
    print(" CUDA DEVICES:")
    if not manifest.cuda_devices:
        print("  None detected.")
    for cuda in manifest.cuda_devices:
        print(f"  [{cuda.device_index}] {cuda.name}")
        print(f"      Compute Capability: {cuda.compute_capability_major}.{cuda.compute_capability_minor} | SMs: {cuda.multiprocessor_count} | Clock: {cuda.clock_rate_mhz} MHz")
        print(f"      VRAM: {cuda.total_memory_mb} MB | LUID: {cuda.luid} | PCI Bus: {cuda.pci_bus_id}")

    print("================================================================================")
    print(" TOPOLOGY EVALUATION & ZERO-COPY VERDICT:")
    print(f"  Verdict:                 {top.topology_type}")
    print(f"  LUID Match:              {top.luid_match}")
    print(f"  Zero-Copy Capture:       {'YES (Capable)' if top.zero_copy_capture_possible else 'NO (Requires Fallback)'}")
    print(f"  Cross-Adapter Fallback:  {'YES (Required)' if top.requires_cross_adapter_fallback else 'NO (Direct)'}")
    print(f"\n  Diagnosis:\n    {top.diagnosis}")
    print("\n  Recommendations:")
    for rec in top.recommendations:
        print(f"    - {rec}")
    print("================================================================================\n")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OpenPrism Hardware & Display/GPU Routing Probe")
    parser.add_argument("--output", "-o", type=str, help="Save machine-readable manifest JSON to path")
    parser.add_argument("--json-only", action="store_true", help="Print JSON manifest directly to stdout")
    parser.add_argument("--check-zero-copy", action="store_true", help="Exit with code 0 only if zero-copy same-adapter capture is possible")
    args = parser.parse_args()

    manifest = collect_environment_manifest()

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())
        print(f"Environment manifest saved to: {args.output}")

    if args.json_only:
        print(manifest.to_json())
    else:
        print_manifest_summary(manifest)

    if args.check_zero_copy:
        return 0 if manifest.topology.zero_copy_capture_possible else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
