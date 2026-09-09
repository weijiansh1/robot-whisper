"""A bounded JSON-lines adapter for persistent, benchmark-specific simulators."""

import json
import queue
import subprocess
import threading
import time


class PersistentWorker:
    def __init__(self, command, env, cwd, log):
        self.log = log
        self.responses = queue.Queue()
        self.process = subprocess.Popen(command, env=env, cwd=cwd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith("COLLECTION_RESULT "):
                    self.responses.put(json.loads(line[len("COLLECTION_RESULT "):]))
                else:
                    self.log.write(line)
                    self.log.flush()
        except Exception as error:
            self.responses.put(error)

    def execute(self, message, timeout, stop):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            if stop.is_set() or time.monotonic() > deadline:
                raise TimeoutError("Persistent environment task stopped or timed out")
            try:
                result = self.responses.get(timeout=.5)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError("Persistent environment exited: %s" % self.process.returncode)
                continue
            if isinstance(result, Exception):
                raise result
            if result["variant"] != message["variant"] or result["pid"] != self.process.pid:
                raise RuntimeError("Persistent environment response identity mismatch")
            return result

    def close(self):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.process.stdout.close()
        self.log.close()
