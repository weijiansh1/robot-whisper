# Overleaf 同步快速参考

## 项目信息

| 项目 | 值 |
|------|-----|
| 主文件 | `main.tex` |
| 本地路径 | `<PROJECT_ROOT>/tex/` |
| 远程仓库 | `https://git.overleaf.com/<YOUR_PROJECT_ID>` |
| 用户名 | `git` |
| 密码 | Overleaf Authentication Token |

> 说明：`<PROJECT_ROOT>` 为本项目根目录；`<YOUR_PROJECT_ID>` 为你的 Overleaf 项目 ID。
> 当前本地仓库已断开 origin 远程，使用前需先接上：
> `cd <PROJECT_ROOT>/tex && git remote add origin https://git@git.overleaf.com/<YOUR_PROJECT_ID>`

---

## 常用命令

### 下载最新（从 Overleaf 拉取）
```bash
cd <PROJECT_ROOT>/tex && git pull origin master
```

### 上传修改（推送到 Overleaf）
```bash
cd <PROJECT_ROOT>/tex && git add -A && git commit -m "更新内容" && git push origin master
```

### 一键同步（先拉后推）
```bash
cd <PROJECT_ROOT>/tex && git pull origin master && git add -A && git commit -m "Sync $(date '+%m-%d %H:%M')" && git push origin master
```

---

## 查看状态

```bash
# 查看本地修改
cd <PROJECT_ROOT>/tex && git status

# 查看最近提交
cd <PROJECT_ROOT>/tex && git log --oneline -5

# 查看远程信息
cd <PROJECT_ROOT>/tex && git remote -v
```

---

## 启动预览服务

```bash
cd <PROJECT_ROOT> && npm run dev
```

访问地址：`http://localhost:<PORT>`（默认 3000，见 `.env` / `next.config.js`）

---

## 工作流程

1. **开始编辑前** → 先拉取最新：`git pull origin master`
2. **本地编辑** → 保存文件，预览会自动刷新
3. **完成后** → 推送到 Overleaf：`git add -A && git commit -m "msg" && git push`
4. **Overleaf 有更新** → 拉取：`git pull origin master`

---

## 冲突处理

如果推送失败（远程有更新）：
```bash
git pull origin master   # 先拉取
# 如有冲突，手动编辑解决
git add -A
git commit -m "Resolve conflicts"
git push origin master
```
