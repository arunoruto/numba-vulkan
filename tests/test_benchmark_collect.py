"""Machine description in benchmarks/collect.py, also for macOS."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "benchmarks"))

import collect


def test_default_machine_label_is_a_clean_host_name(monkeypatch):
    monkeypatch.setattr(collect.socket, "gethostname", lambda: "Mirza's-MacBook.local")
    assert collect._default_machine() == "Mirza-s-MacBook"
    assert collect.NAME.match(collect._default_machine())


def test_macos_description(monkeypatch):
    values = {"machdep.cpu.brand_string": "Apple M2 Pro", "hw.memsize": "17179869184"}
    monkeypatch.setattr(collect.sys, "platform", "darwin")
    monkeypatch.setattr(collect, "_sysctl", values.get)
    monkeypatch.setattr(
        collect.platform, "mac_ver", lambda: ("15.1", ("", "", ""), "arm64")
    )
    monkeypatch.setattr(collect.platform, "machine", lambda: "arm64")
    assert collect._cpu_model() == "Apple M2 Pro"
    assert collect._memory_gib() == 16.0
    assert collect._os_name() == "macOS 15.1 (arm64)"


def test_macos_description_without_sysctl(monkeypatch):
    monkeypatch.setattr(collect.sys, "platform", "darwin")
    monkeypatch.setattr(collect, "_sysctl", lambda name: None)
    monkeypatch.setattr(collect.platform, "machine", lambda: "arm64")
    assert collect._cpu_model() == "arm64"
    assert collect._memory_gib() is None
