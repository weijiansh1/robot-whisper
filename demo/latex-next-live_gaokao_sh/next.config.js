/** @type {import('next').NextConfig} */
// 反向代理子路径支持（如 JupyterHub server-proxy: /user/<name>/proxy/<port>）。
// 设了 APP_BASE_PATH 时：静态资源 URL 带上前缀；同时把它注入到客户端代码。
// 不设时为空字符串，行为与原来完全一致（本地直接访问 :2219 不受影响）。
const BASE = (process.env.APP_BASE_PATH || '').replace(/\/$/, '');
const nextConfig = {
  reactStrictMode: false,
  assetPrefix: BASE || undefined,
  env: {
    NEXT_PUBLIC_APP_BASE_PATH: BASE,
    NEXT_PUBLIC_CLAUDE_MODEL: process.env.CLAUDE_MODEL || '',  // 空=用账号默认模型
  },
  // 强制 Node 运行时，便于子进程/文件监听
  experimental: { serverActions: { allowedOrigins: ['localhost'] } }
};
module.exports = nextConfig;
