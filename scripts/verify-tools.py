"""Test tools separately from GPU checks to limit concurrent memory use."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request


def notebook():
    from jupyter_client import KernelManager

    manager = KernelManager(kernel_name="python3", transport="ipc")
    manager.start_kernel()
    client = manager.client()
    try:
        client.start_channels()
        client.wait_for_ready(timeout=60)
        msg_id = client.execute(
            "import torch; assert torch.cuda.is_available(); "
            "assert torch.ones(1, device='cuda').item() == 1"
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            reply = client.get_shell_msg(timeout=60)
            if reply["parent_header"].get("msg_id") == msg_id:
                assert reply["content"]["status"] == "ok", reply["content"]
                break
        else:
            raise RuntimeError("Notebook kernel timed out")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    print("PASS notebook kernel GPU execution", flush=True)

    with tempfile.TemporaryDirectory() as directory:
        token = os.urandom(24).hex()
        with open(Path(directory) / "jupyter.log", "w+") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "jupyterlab", "--no-browser", "--allow-root",
                 "--ip=127.0.0.1", "--port=8899", "--ServerApp.port_retries=0"],
                env=dict(os.environ, JUPYTER_TOKEN=token), stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("Jupyter server exited before becoming ready")
                    try:
                        request = urllib.request.Request(
                            "http://127.0.0.1:8899/api/status",
                            headers={"Authorization": f"token {token}"},
                        )
                        with urllib.request.urlopen(request, timeout=2) as response:
                            assert response.status == 200
                        break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise RuntimeError("Jupyter HTTP readiness timed out")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print("PASS authenticated Jupyter HTTP response", flush=True)


def wandb_offline():
    import wandb
    with tempfile.TemporaryDirectory() as directory:
        with wandb.init(project="container-smoke", mode="offline", dir=directory) as run:
            run.log({"smoke": 1})
    print("PASS W&B offline logging")


def monitor():
    from jtop import jtop
    with jtop() as jetson:
        assert jetson.ok()
        assert jetson.gpu and jetson.memory["RAM"]
        print("PASS jtop connection and raw GPU/RAM data")
        try:
            assert jetson.stats
            print("PASS compact stats API")
        except KeyError as error:
            # Report the one previously reproduced upstream schema defect explicitly.
            if error.args != ("online",):
                raise
            print("KNOWN LIMITATION: compact stats API lacks an engine 'online' field")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("notebook", "wandb", "jtop"))
    args = parser.parse_args()
    {"notebook": notebook, "wandb": wandb_offline, "jtop": monitor}[args.tool]()
