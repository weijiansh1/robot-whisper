# 远程 /data 环境在本机的对应设置（对应远程 /root/.bashrc 的自定义部分）
# 用法: source ~/data/env.sh
export DATA=/home/swj/data
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
# conda (torch 环境)
if [ -f "$DATA/miniconda/etc/profile.d/conda.sh" ]; then
  . "$DATA/miniconda/etc/profile.d/conda.sh"
  conda activate torch 2>/dev/null
fi
# nvm / node 20 / codex
export NVM_DIR="$DATA/nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
export PATH="$DATA/npm-global/bin:$PATH"
# OPENAI_* 等凭据从同步下来的远程 .bashrc 里读取，不在这里明文保存
if [ -f "$DATA/_root/.bashrc" ]; then
  eval "$(grep -E '^export (OPENAI_|HF_|HUGGINGFACE|WANDB)' "$DATA/_root/.bashrc")"
fi
# 镜像自带的 LD_LIBRARY_PATH 含系统 python3.12 的 torch 库目录，会污染其它 Python 的 torch，去掉
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE '/dist-packages/torch(_tensorrt)?/lib' | paste -sd: -)
# 无 root 权限下补装的 OpenGL/EGL/OSMesa 运行库（来自 apt 包解包，见 ~/.local/syslib）
export LD_LIBRARY_PATH="$HOME/.local/syslib/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_DIRS="$HOME/.local/syslib/glvnd/egl_vendor.d"
export LIBGL_DRIVERS_PATH="$HOME/.local/syslib/usr/lib/x86_64-linux-gnu/dri"
# ImageMagick 6（LIBERO-Plus 的 Wand 依赖），同样是 apt 包解包到 ~/.local/syslib
export MAGICK_HOME=/home/swj/.local/magick
export MAGICK_CODER_MODULE_PATH=/home/swj/.local/syslib/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/coders
export MAGICK_CODER_FILTER_PATH=/home/swj/.local/syslib/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.12/modules-Q16/filters
export MAGICK_CONFIGURE_PATH=/home/swj/.local/syslib/etc/ImageMagick-6:/home/swj/.local/syslib/usr/share/ImageMagick-6
