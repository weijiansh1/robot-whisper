#!/usr/bin/env bash
# 把从远程原样同步过来的 uv/venv 虚拟环境迁移到新路径
# 用法: relocate-venv.sh <venv目录> <旧前缀> <新前缀> <python可执行目录(新home)>
set -euo pipefail
VENV=$1; OLD=$2; NEW=$3; HOME_BIN=$4
[ -f "$VENV/pyvenv.cfg" ] || { echo "no pyvenv.cfg in $VENV" >&2; exit 1; }
# 1. pyvenv.cfg home
sed -i "s|^home = .*|home = $HOME_BIN|" "$VENV/pyvenv.cfg"
# 2. bin/python* 软链接
PYVER=$("$HOME_BIN/python3" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
cd "$VENV/bin"
for l in python python3 "python$PYVER"; do rm -f "$l"; done
ln -s "$HOME_BIN/python$PYVER" "python$PYVER"; ln -s "python$PYVER" python3; ln -s python3 python
# 3. shebang 与其他文本文件里的旧前缀
grep -rIl --exclude-dir=__pycache__ -- "$OLD" "$VENV/bin" "$VENV/lib" 2>/dev/null | while read -r f; do
  sed -i "s|$OLD|$NEW|g" "$f"
done
echo "relocated $VENV -> home=$HOME_BIN"
"$VENV/bin/python" -c "import sys;print('python', sys.version.split()[0], 'prefix', sys.prefix)"
