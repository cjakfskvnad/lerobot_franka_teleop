#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

import yaml
from oculus_reader import OculusReader


def load_oculus_ip(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        return cfg["record"]["teleop"]["oculus_config"]["ip"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not read Oculus IP from {config_path}") from exc


def ensure_adb_connected(ip_address: str) -> None:
    serial = f"{ip_address}:5555"
    subprocess.run(["adb", "connect", serial], check=False)
    state = subprocess.run(
        ["adb", "-s", serial, "get-state"],
        check=False,
        capture_output=True,
        text=True,
    )
    if state.stdout.strip() != "device":
        raise RuntimeError(
            f"Oculus wireless ADB is not accessible at {serial}. "
            "Wake the headset, check Wi-Fi, or reconnect once over USB with `adb tcpip 5555`."
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_config = repo_root / "scripts" / "config" / "record_cfg.yaml"

    parser = argparse.ArgumentParser(description="Test Oculus Reader over wireless ADB.")
    parser.add_argument("--ip", help="Oculus Quest IP address. Overrides record_cfg.yaml.")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to record_cfg.yaml.",
    )
    parser.add_argument("--period", type=float, default=0.3, help="Print interval in seconds.")
    args = parser.parse_args()

    ip_address = args.ip or load_oculus_ip(args.config)
    print(f"Connecting to Oculus Reader over wireless ADB: {ip_address}:5555")
    print("Press Ctrl-C to stop.")

    ensure_adb_connected(ip_address)
    reader = OculusReader(ip_address=ip_address)

    try:
        while True:
            time.sleep(args.period)
            transforms, buttons = reader.get_transformations_and_buttons()
            print(transforms, buttons)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
