import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / "venv" / "bin" / "python"


def find_free_port(start=8011, end=8999, excluded=()):
    for port in range(start, end + 1):
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available in the configured range")


def wait_for_http(url, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1].split("/")[0])), timeout=1):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def main():
    risk_port = find_free_port(8011, 8020)
    policy_port = find_free_port(8011, 8020, excluded={risk_port})

    env = os.environ.copy()
    env["RISK_MCP_PORT"] = str(risk_port)
    env["POLICY_MCP_PORT"] = str(policy_port)

    procs = [
        subprocess.Popen([str(VENV_PYTHON), "insurance_risk_server.py"], cwd=str(ROOT), env=env),
        subprocess.Popen([str(VENV_PYTHON), "insurance_policy_server.py"], cwd=str(ROOT), env=env),
    ]

    try:
        print(f"Starting Insurance MCP services on ports {risk_port} and {policy_port}...")
        if not wait_for_http(f"http://127.0.0.1:{risk_port}/mcp"):
            raise RuntimeError("Risk MCP server did not become ready")
        if not wait_for_http(f"http://127.0.0.1:{policy_port}/mcp"):
            raise RuntimeError("Policy MCP server did not become ready")

        print("Insurance MCP services are running.")
        print(f"RISK_MCP_PORT={risk_port}")
        print(f"POLICY_MCP_PORT={policy_port}")
        print("Press Ctrl+C to stop both services.")

        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("Stopping Insurance MCP services...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    main()
