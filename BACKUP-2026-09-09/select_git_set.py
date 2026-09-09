import os, re, subprocess, collections, sys
OUT=sys.argv[1]
r=subprocess.run(["git","ls-files","-o","--exclude-standard","-z"],capture_output=True)
entries=[p for p in r.stdout.decode("utf-8","surrogateescape").split("\0") if p]
BIG_EXT={".npz",".npy",".gz",".tgz",".tar",".zip",".7z",".h5",".hdf5",".mp4",".avi",".mkv",".bin",".parquet",".arrow"}
LIMIT=95*2**20
inc=[];exc=[]
for p in entries:
    if p.endswith("/"): exc.append(("nested-repo",0,p)); continue
    try: st=os.lstat(p)
    except FileNotFoundError: exc.append(("missing",0,p)); continue
    ext=os.path.splitext(p)[1].lower()
    if ext in BIG_EXT: exc.append(("data-ext",st.st_size,p)); continue
    if st.st_size>LIMIT: exc.append((">95MiB",st.st_size,p)); continue
    inc.append((st.st_size,p))
with open(f"{OUT}/git-include.lst0","w",encoding="utf-8",errors="surrogateescape") as f: f.write("\0".join(p for _,p in inc))
with open(f"{OUT}/git-exclude.tsv","w",encoding="utf-8",errors="surrogateescape") as f:
    for why,s,p in exc: f.write(f"{why}\t{s}\t{p}\n")
by=collections.defaultdict(lambda:[0,0])
for s,p in inc:
    k=p.split("/")[0]; by[k][0]+=s; by[k][1]+=1
tot=sum(s for s,_ in inc)
print(f"INCLUDE {len(inc)} files {tot/2**30:.2f} GiB ; EXCLUDE {len(exc)} entries {sum(s for _,s,_ in exc)/2**30:.2f} GiB")
print("-- include by top dir (GiB, files) --")
for k,(s,n) in sorted(by.items(),key=lambda kv:-kv[1][0])[:25]: print(f"{s/2**30:7.2f}  {n:7d}  {k}")
print("-- excluded by reason --")
c=collections.defaultdict(lambda:[0,0])
for why,s,p in exc: c[why][0]+=s; c[why][1]+=1
for k,(s,n) in c.items(): print(f"{s/2**30:7.2f}  {n:7d}  {k}")
