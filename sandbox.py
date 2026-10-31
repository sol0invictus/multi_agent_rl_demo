"""Docker sandbox: one long-lived container (no network, non-root), one workspace dir per rollout."""
import asyncio
import os
import shutil
import subprocess
import uuid

IMAGE = "harness-sandbox:latest"
HOST_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "workspaces")
CT_ROOT = "/workspaces"
MAX_OUT = 2500


class Sandbox:
    def __init__(self, name="org-sbx", max_concurrent=16):
        self.name = name
        self.sem = asyncio.Semaphore(max_concurrent)
        os.makedirs(HOST_ROOT, exist_ok=True)
        os.chmod(HOST_ROOT, 0o777)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "run", "-d", "--name", name, "--network", "none", "--memory", "4g",
                        "--pids-limit", "512", "--cpus", "8", "-v", f"{HOST_ROOT}:{CT_ROOT}", IMAGE,
                        "sleep", "infinity"], check=True, capture_output=True)

    def new_workspace(self, files):
        wid = uuid.uuid4().hex[:12]
        host = os.path.join(HOST_ROOT, wid)
        os.makedirs(host)
        for fn, text in files.items():   # fn may be "site/orders.csv"
            d = os.path.dirname(os.path.join(host, fn))
            os.makedirs(d, exist_ok=True)
            os.chmod(d, 0o777)
            with open(os.path.join(host, fn), "w") as f:
                f.write(text)
            os.chmod(os.path.join(host, fn), 0o666)
        os.chmod(host, 0o777)
        return Workspace(self, wid, f"{CT_ROOT}/{wid}", host)

    async def exec(self, workdir, argv, timeout=20):
        async with self.sem:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", workdir, self.name, "timeout", "-k", "2", str(timeout), *argv,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout + 10)
            except asyncio.TimeoutError:
                proc.kill()
                return 124, "[timed out]"
        text = out.decode(errors="replace")
        if len(text) > MAX_OUT:
            text = text[:MAX_OUT // 2] + f"\n... [{len(text) - MAX_OUT} chars truncated] ...\n" + text[-MAX_OUT // 2:]
        rc = proc.returncode
        if rc == 124:
            text += "\n[command timed out]"
        return rc, text

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


class Workspace:
    def __init__(self, sbx, wid, path, host):
        self.sbx, self.wid, self.path, self.host = sbx, wid, path, host

    async def bash(self, cmd, timeout=20):
        return await self.sbx.exec(self.path, ["bash", "-c", cmd], timeout)

    async def argv(self, argv, timeout=20):
        return await self.sbx.exec(self.path, list(argv), timeout)

    def read_host(self, fn):
        p = os.path.join(self.host, fn)
        if not os.path.isfile(p):
            return None
        with open(p, errors="replace") as f:
            return f.read()

    def cleanup(self):
        shutil.rmtree(self.host, ignore_errors=True)
