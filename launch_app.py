"""Launch the Streamlit prototype locally, including from a packaged launcher."""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


APP_ROOT = Path(
    sys.executable if getattr(sys, "frozen", False) else __file__
).resolve().parent
APP_FILE = APP_ROOT / "prototype_app.py"


def find_python():
    candidates = [
        APP_ROOT / ".venv" / "Scripts" / "python.exe",
        APP_ROOT / ".venv-1" / "Scripts" / "python.exe",
    ]
    if not getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable))

    for candidate in candidates:
        if candidate.is_file() and subprocess.run(
            [str(candidate), "-c", "import streamlit"],
            cwd=APP_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return str(candidate)

    for command in (["py", "-3"], ["python"]):
        executable = shutil.which(command[0])
        if executable and subprocess.run(
            [executable, *command[1:], "-c", "import streamlit"],
            cwd=APP_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return [executable, *command[1:]]

    raise RuntimeError(
        "Python with the project dependencies is not installed. "
        "Run setup.bat or install requirements.txt, then start this launcher again."
    )


def find_port():
    for port in range(8501, 8511):
        with socket.socket() as server_socket:
            try:
                server_socket.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free localhost port was found between 8501 and 8510.")


def main():
    if not APP_FILE.is_file():
        raise RuntimeError(
            "prototype_app.py was not found. Keep this launcher beside the project source files."
        )

    python = find_python()
    command = python if isinstance(python, list) else [python]
    port = find_port()
    url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            *command,
            "-m",
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        cwd=APP_ROOT,
    )
    print(f"Starting the G65 Phishing Detection app at {url}")
    print("Keep this window open while using the app. Press Ctrl+C to stop it.")

    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Streamlit exited during startup with code {process.returncode}."
                )
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    if response.status == 200:
                        webbrowser.open(url)
                        print(f"App is running: {url}")
                        process.wait()
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(0.5)
        raise RuntimeError("The app did not become ready within 60 seconds.")
    except KeyboardInterrupt:
        print("\nStopping the local app...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        input("Press Enter to close this window...")
        raise SystemExit(1)
