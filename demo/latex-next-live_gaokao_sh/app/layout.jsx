export const metadata = {
  title: 'LaTeX Live Preview',
  description: 'Live LaTeX preview with reverse TeX lookup',
}

export default function RootLayout({ children }) {
  return (
    <html lang="zh">
      <body>{children}</body>
    </html>
  )
}