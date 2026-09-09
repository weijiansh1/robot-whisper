import os, sys, hashlib, subprocess, re, time
from multiprocessing import Pool
ROOT = "/home/jovyan/work/himoe-vla"
OUT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

def git_list(extra):
    r = subprocess.run(["git","ls-files","-o","--exclude-standard","-z"]+extra, capture_output=True)
    return [p for p in r.stdout.decode("utf-8","surrogateescape").split("\0") if p]

untracked = git_list([])
ignored = git_list(["-i"])

EXCL = re.compile(r"(^|/)(uv-python|envs|\.venv|venv|node_modules|__pycache__|\.cache|\.next|\.pytest_cache|\.ruff_cache|\.ipynb_checkpoints)(/|$)")
EXT_EXCL = (".pth",".pt",".safetensors",".ckpt",".pyc",".pyo",".log",".pid")
KEEP_CACHE = ("himoe-vla-cache/himoe-libero-bridge/formal-artifacts/",
              "himoe-vla-cache/himoe-libero-bridge/paper-alignment/",
              "himoe-vla-cache/himoe-libero-bridge/metadata/",
              "himoe-vla-cache/himoe-libero-bridge/moevla-data/",
              "himoe-vla-cache/vla-adapter-repro/results/")
def keep_ignored(p):
    if EXCL.search(p) or p.endswith(EXT_EXCL): return False
    if p.startswith("himoe-vla-cache/"):
        return p.startswith(KEEP_CACHE)
    return True

items = [(p,"untracked") for p in untracked] + [(p,"ignored") for p in ignored if keep_ignored(p)]
skipped = [p for p in ignored if not keep_ignored(p)]
with open(os.path.join(OUT,"skipped-ignored.lst"),"w",encoding="utf-8",errors="surrogateescape") as f:
    f.write("\n".join(skipped)+"\n")

def h(item):
    p, tier = item
    try:
        st = os.lstat(p)
        if os.path.islink(p):
            return (tier, "SYMLINK", st.st_size, int(st.st_mtime), p, os.readlink(p))
        if not os.path.isfile(p):
            return (tier, "NOTFILE", 0, 0, p, "")
        m = hashlib.sha256()
        with open(p,"rb",buffering=0) as fh:
            for chunk in iter(lambda: fh.read(8<<20), b""):
                m.update(chunk)
        return (tier, m.hexdigest(), st.st_size, int(st.st_mtime), p, "")
    except Exception as e:
        return (tier, "ERROR", 0, 0, p, repr(e))

t0=time.time(); n=0; tot=0
with open(os.path.join(OUT,"manifest-raw-data.tsv"),"w",encoding="utf-8",errors="surrogateescape") as f, Pool(4) as pool:
    f.write("tier\tsha256\tbytes\tmtime\tpath\tnote\n")
    for rec in pool.imap_unordered(h, items, chunksize=64):
        f.write("\t".join(str(x) for x in rec)+"\n")
        n+=1; tot+=rec[2]
        if n%5000==0:
            print(f"{n}/{len(items)} files, {tot/2**30:.1f} GiB, {time.time()-t0:.0f}s", flush=True)
print(f"DONE {n} files, {tot/2**30:.1f} GiB, {time.time()-t0:.0f}s", flush=True)
