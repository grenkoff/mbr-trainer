"""CPU temperature guard for long solver runs: a laptop running all cores flat out for hours can overheat
and switch itself off."""

import os
import signal
import time
from pathlib import Path


def cpu_temperature():
    """The CPU package temperature in °C (coretemp), or None if there is no such sensor."""
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() != "coretemp":
                continue
            for label in hwmon.glob("temp*_label"):
                if label.read_text().startswith("Package"):
                    return int(label.with_name(label.name.replace("label", "input")).read_text()) / 1000
        except OSError:
            continue
    return None


def guard_temperature(pids, max_temp, resume_temp, stop, on_change=None):
    """Pause the processes pids() returns (SIGSTOP) above max_temp, resume them (SIGCONT) at resume_temp,
    until stop is set; on_change(paused, temperature) is told of every switch. They are never left paused."""
    paused = False
    try:
        while not stop.is_set():
            t = cpu_temperature()
            if t is not None and (not paused and t >= max_temp or paused and t <= resume_temp):
                paused = not paused
                _signal(pids(), signal.SIGSTOP if paused else signal.SIGCONT)
                if on_change:
                    on_change(paused, t)
            stop.wait(2)
    finally:
        if paused:
            _signal(pids(), signal.SIGCONT)


def _signal(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def cool_down(max_temp=80, resume_temp=75, log=None):
    """Wait while the CPU is hotter than max_temp, until it is down to resume_temp."""
    t = cpu_temperature()
    if t is None or t < max_temp:
        return
    if log:
        log(f"CPU {t:.0f}°C: waiting until {resume_temp}°C")
    while (t := cpu_temperature()) is not None and t > resume_temp:
        time.sleep(2)
