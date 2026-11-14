"""Small Windows-only resource sampler and bounded child-process job."""

import ctypes as ct
from ctypes import wintypes as wt
import os
from pathlib import Path
import subprocess
import time

if os.name != "nt":
    raise RuntimeError("The desktop profiler requires native Windows")

import _winapi
import msvcrt

KERNEL = ct.WinDLL("kernel32", use_last_error=True)
SIZE = ct.c_size_t
U64 = ct.c_ulonglong


class MemoryStatus(ct.Structure):
    _fields_ = [("length", wt.DWORD), ("load", wt.DWORD)] + [
        (name, U64) for name in ("total_phys", "avail_phys", "total_pagefile", "avail_pagefile",
                                "total_virtual", "avail_virtual", "avail_extended")]


class BasicLimits(ct.Structure):
    _fields_ = [("process_time", ct.c_longlong), ("job_time", ct.c_longlong),
                ("flags", wt.DWORD), ("min_ws", SIZE), ("max_ws", SIZE),
                ("active_limit", wt.DWORD), ("affinity", SIZE),
                ("priority", wt.DWORD), ("scheduling", wt.DWORD)]


class IoCounters(ct.Structure):
    _fields_ = [(name, U64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ct.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", SIZE),
                ("job_memory", SIZE), ("peak_process_memory", SIZE), ("peak_job_memory", SIZE)]


class Accounting(ct.Structure):
    _fields_ = [(name, ct.c_longlong) for name in ("user_time", "kernel_time", "period_user", "period_kernel")] + [
        (name, wt.DWORD) for name in ("page_faults", "total_processes", "active_processes", "terminated_processes")]


class MemoryCounters(ct.Structure):
    _fields_ = [("cb", wt.DWORD), ("page_faults", wt.DWORD)] + [(name, SIZE) for name in (
        "peak_ws", "ws", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]


def api(name, restype, *args):
    function = getattr(KERNEL, name)
    function.restype, function.argtypes = restype, args
    return function


create_job = api("CreateJobObjectW", wt.HANDLE, ct.c_void_p, wt.LPCWSTR)
set_job = api("SetInformationJobObject", wt.BOOL, wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD)
query_job = api("QueryInformationJobObject", wt.BOOL, wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD, ct.c_void_p)
assign_job = api("AssignProcessToJobObject", wt.BOOL, wt.HANDLE, wt.HANDLE)
terminate_job = api("TerminateJobObject", wt.BOOL, wt.HANDLE, wt.UINT)
close_handle = api("CloseHandle", wt.BOOL, wt.HANDLE)
resume_thread = api("ResumeThread", wt.DWORD, wt.HANDLE)
terminate_process = api("TerminateProcess", wt.BOOL, wt.HANDLE, wt.UINT)
get_current = api("GetCurrentProcess", wt.HANDLE)
get_affinity = api("GetProcessAffinityMask", wt.BOOL, wt.HANDLE, ct.POINTER(SIZE), ct.POINTER(SIZE))
get_memory = api("GlobalMemoryStatusEx", wt.BOOL, ct.POINTER(MemoryStatus))
get_system_times = api("GetSystemTimes", wt.BOOL, ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME), ct.POINTER(wt.FILETIME))
open_process = api("OpenProcess", wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
get_process_memory = api("K32GetProcessMemoryInfo", wt.BOOL, wt.HANDLE, ct.POINTER(MemoryCounters), wt.DWORD)
get_image = api("QueryFullProcessImageNameW", wt.BOOL, wt.HANDLE, wt.DWORD, wt.LPWSTR, ct.POINTER(wt.DWORD))
get_exit = api("GetExitCodeProcess", wt.BOOL, wt.HANDLE, ct.POINTER(wt.DWORD))
wait_process = api("WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD)


def check(result):
    if not result:
        raise ct.WinError(ct.get_last_error())
    return result


def system_sample():
    memory = MemoryStatus(length=ct.sizeof(MemoryStatus))
    check(get_memory(ct.byref(memory)))
    times = [wt.FILETIME() for _ in range(3)]
    check(get_system_times(*(ct.byref(value) for value in times)))
    idle, kernel, user = [(value.dwHighDateTime << 32) + value.dwLowDateTime for value in times]
    return {"ram_total_bytes": memory.total_phys, "ram_available_bytes": memory.avail_phys,
            "commit_headroom_bytes": memory.avail_pagefile, "cpu_idle_ticks": idle,
            "cpu_total_ticks": kernel + user}


class ProcessJob:
    def __init__(self, cores, memory_bytes):
        self.handle, self.process = None, None
        self.pid = None
        available, system = SIZE(), SIZE()
        check(get_affinity(get_current(), ct.byref(available), ct.byref(system)))
        bits = [1 << i for i in range(64) if available.value & (1 << i)]
        if len(bits) < cores:
            raise RuntimeError("Not enough CPUs in the current process affinity mask")
        self.affinity_mask = sum(bits[:cores])
        self.handle = check(create_job(None, None))
        limits = ExtendedLimits()
        # Entire process tree: affinity, below-normal priority, commit limit,
        # and automatic termination when the monitor's last job handle closes.
        limits.basic.flags = 0x10 | 0x20 | 0x200 | 0x2000
        limits.basic.affinity = self.affinity_mask
        limits.basic.priority = 0x4000
        limits.job_memory = memory_bytes
        try:
            check(set_job(self.handle, 9, ct.byref(limits), ct.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def start(self, command, cwd, log_path):
        with open(log_path, "wb") as log, open(os.devnull, "rb") as stdin:
            output_handle = msvcrt.get_osfhandle(log.fileno())
            input_handle = msvcrt.get_osfhandle(stdin.fileno())
            os.set_handle_inheritable(output_handle, True)
            os.set_handle_inheritable(input_handle, True)
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = input_handle, output_handle, output_handle
            process, thread, pid, _ = _winapi.CreateProcess(
                command[0], subprocess.list2cmdline(command), None, None, True,
                0x4 | 0x4000 | 0x08000000, None, str(cwd), startup)
            self.process, self.pid = process, pid
            try:
                # The child cannot execute, spawn workers or allocate its main
                # working set until all job limits have been attached.
                check(assign_job(self.handle, process))
                if resume_thread(thread) == 0xFFFFFFFF:
                    raise ct.WinError(ct.get_last_error())
            except BaseException:
                terminate_process(process, 125)
                raise
            finally:
                close_handle(thread)

    def sample(self):
        accounting, limits = Accounting(), ExtendedLimits()
        check(query_job(self.handle, 1, ct.byref(accounting), ct.sizeof(accounting), None))
        check(query_job(self.handle, 9, ct.byref(limits), ct.sizeof(limits), None))
        buffer = ct.create_string_buffer(8 + 256 * ct.sizeof(SIZE))
        check(query_job(self.handle, 3, buffer, len(buffer), None))
        count = ct.cast(ct.addressof(buffer) + 4, ct.POINTER(wt.DWORD))[0]
        pids = ct.cast(ct.addressof(buffer) + 8, ct.POINTER(SIZE))
        processes, query_errors = [], []
        for pid in list(pids[:count]):
            handle = open_process(0x1000 | 0x10, False, pid)
            if not handle:
                continue  # A short-lived worker may have exited between queries.
            try:
                memory = MemoryCounters(cb=ct.sizeof(MemoryCounters))
                check(get_process_memory(handle, ct.byref(memory), ct.sizeof(memory)))
                name, length = ct.create_unicode_buffer(32768), wt.DWORD(32768)
                check(get_image(handle, 0, name, ct.byref(length)))
                affinity, system = SIZE(), SIZE()
                check(get_affinity(handle, ct.byref(affinity), ct.byref(system)))
                processes.append({"pid": pid, "name": Path(name.value).name, "private_bytes": memory.private,
                                  "working_set_bytes": memory.ws, "affinity_mask": affinity.value})
            except OSError as error:
                # A process can exit after enumeration but before metadata reads.
                # Kernel job accounting/limits remain available for the whole tree.
                query_errors.append({"pid": pid, "error": str(error)})
            finally:
                close_handle(handle)
        exit_code = wt.DWORD()
        check(get_exit(self.process, ct.byref(exit_code)))
        return {"processes": processes, "process_query_errors": query_errors, "active_processes": accounting.active_processes,
                "cpu_ticks": accounting.user_time + accounting.kernel_time,
                "private_bytes_sum": sum(p["private_bytes"] for p in processes),
                "working_set_bytes_sum": sum(p["working_set_bytes"] for p in processes),
                "peak_job_commit_bytes": limits.peak_job_memory,
                "root_exit_code": exit_code.value}

    def stop(self):
        if self.handle:
            check(terminate_job(self.handle, 124))

    def close(self):
        if self.handle:
            close_handle(self.handle)
            self.handle = None
        if self.process:
            close_handle(self.process)
            self.process = None


def verify_job(output):
    """Exercise actual child-tree timeout and memory denial before launching UE."""
    import sys
    output.mkdir(parents=True, exist_ok=True)
    job = ProcessJob(2, 256 * 1024**2)
    try:
        command = [sys.executable, "-c", "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print('CHILD_PID',p.pid,flush=True); time.sleep(60)"]
        job.start(command, output, output / "job-selftest.log")
        for _ in range(50):
            before = job.sample()
            marker = (output / "job-selftest.log").read_text()
            if "CHILD_PID" in marker:
                child_pid = int(marker.split("CHILD_PID")[1].split()[0])
                if child_pid in [p["pid"] for p in before["processes"]]:
                    break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"Child-tree selftest did not observe the spawned child: {before}")
        if any(p["affinity_mask"] != job.affinity_mask for p in before["processes"]):
            raise RuntimeError("Job affinity was not inherited")
        job.stop()
        for _ in range(30):
            after = job.sample()
            if after["active_processes"] == 0:
                break
            time.sleep(0.1)
        if after["active_processes"]:
            raise RuntimeError("Job did not terminate its child tree")
    finally:
        job.close()
    limited = ProcessJob(2, 128 * 1024**2)
    try:
        command = [sys.executable, "-c", "x=bytearray(512*1024**2)"]
        limited.start(command, output, output / "job-memory-selftest.log")
        for _ in range(50):
            state = limited.sample()
            if not state["active_processes"]:
                break
            time.sleep(0.1)
        if state["active_processes"] or state["root_exit_code"] == 0:
            raise RuntimeError("Job memory limit was not enforced")
        if "MemoryError" not in (output / "job-memory-selftest.log").read_text():
            raise RuntimeError("Memory-limit child failed for a different reason")
    finally:
        limited.close()
    closing = ProcessJob(2, 256 * 1024**2)
    child_handle = None
    try:
        command = [sys.executable, "-c", "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print('CHILD_PID',p.pid,flush=True); time.sleep(60)"]
        closing.start(command, output, output / "job-close-selftest.log")
        for _ in range(50):
            marker = (output / "job-close-selftest.log").read_text()
            if "CHILD_PID" in marker:
                child_pid = int(marker.split("CHILD_PID")[1].split()[0])
                child_handle = check(open_process(0x100000, False, child_pid))
                break
            time.sleep(0.1)
        if not child_handle:
            raise RuntimeError("Close-on-exit selftest child did not start")
        closing.close()  # No explicit TerminateJobObject call in this scenario.
        if wait_process(child_handle, 3000) != 0:
            raise RuntimeError("Closing the job handle did not terminate its child")
    finally:
        closing.close()
        if child_handle:
            close_handle(child_handle)
    return {"child_tree_terminated": True, "inherited_affinity": before["processes"],
            "memory_allocation_denied": True, "kill_on_close_terminated_child": True}
