"""A private MPS controller restricted to one explicitly allowed physical GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)


class PrivateMPS:
    def __init__(self, gpu, output):
        if gpu not in ALLOWED_GPUS:
            raise ValueError("MPS cannot use physical GPU 6")
        self.uuid = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
            "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        if not self.uuid.startswith("GPU-") or "\n" in self.uuid:
            raise ValueError("MPS requires exactly one physical GPU UUID")
        self.pipe = Path(tempfile.mkdtemp(prefix="moe-control-mps-"))
        self.log_directory = (Path(output) / "mps").resolve()
        self.log_directory.mkdir(parents=True, exist_ok=False)
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_MPS_PIPE_DIRECTORY=str(self.pipe), CUDA_MPS_LOG_DIRECTORY=str(self.log_directory))
        self.log = (self.log_directory / "controller_console.log").open("w")
        self.process = None
        self.server_pids = []

    def command(self, text):
        result = subprocess.run(["nvidia-cuda-mps-control"], input=text + "\n", env=self.env,
            text=True, capture_output=True, timeout=15, check=True)
        return result.stdout.strip()

    def start(self):
        self.process = subprocess.Popen(["nvidia-cuda-mps-control", "-f"], env=self.env,
            stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while not (self.pipe / "nvidia-cuda-mps-control.pid").exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Private MPS controller did not start")
            time.sleep(.2)
        self.command("get_server_list")
        return dict(gpu_uuid=self.uuid, pipe_directory=str(self.pipe), controller_pid=self.process.pid)

    def verify_clients(self, expected):
        self.server_pids = [int(pid) for pid in self.command("get_server_list").split()]
        if len(self.server_pids) != 1:
            raise ValueError("Expected one private MPS server")
        clients = [int(pid) for pid in self.command("get_client_list %d" % self.server_pids[0]).split()]
        if set(clients) != set(expected):
            raise ValueError("MPS client identity mismatch: %s versus %s" % (clients, expected))
        return dict(server_pids=self.server_pids, client_pids=clients,
                    device_client_list=self.command("get_device_client_list"))

    def close(self):
        result = dict(controller_stopped=True, live_server_pids=[])
        try:
            if self.process is not None and self.process.poll() is None:
                result["quit_response"] = self.command("quit")
                self.process.wait(timeout=20)
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            result["controller_stopped"] = self.process is None or self.process.poll() is not None
            result["live_server_pids"] = [pid for pid in self.server_pids if Path("/proc/%d" % pid).exists()]
            self.log.close()
            if result["controller_stopped"] and not result["live_server_pids"]:
                shutil.rmtree(self.pipe)
        return result
