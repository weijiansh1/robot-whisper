#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, json, subprocess, re, os

LIGATURES = {
  '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl',
  '\ufb05': 'ft', '\ufb06': 'st'
}

def norm_text(s: str) -> str:
  if not s: return ''
  # 去零宽/控制符
  s = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', s)
  # 连字规范
  for k,v in LIGATURES.items():
    s = s.replace(k, v)
  # 行尾断词：confi-\n guration -> configuration
  s = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', s)
  # 普通换行 -> 单空格；多空格压缩
  s = re.sub(r'\s*\n\s*', ' ', s)
  s = re.sub(r'\s{2,}', ' ', s)
  return s.strip()

def read_context(file, line):
  try:
    with open(file, 'r', encoding='utf-8') as f:
      arr = f.read().splitlines()
    s = max(1, line-2); e = min(len(arr), line+2)
    return "\n".join(f"{i}: {arr[i-1]}" for i in range(s, e+1))
  except:
    return ""

def synctex_query(pdf, page, h, v):
  try:
    out = subprocess.check_output(['synctex','edit','-o',f'{page}:{int(h)}:{int(v)}:{pdf}'], text=True)
    f = re.search(r'Input:\s*(.+\.tex)', out)
    ln = re.search(r'Line:\s*(\d+)', out)
    if f and ln:
      file = f.group(1).strip()
      line = int(ln.group(1))
      return {'method':'synctex', 'file': file, 'line': line, 'snippet': read_context(file, line)}
  except Exception:
    return None

def rg_candidates(tex_root, out_dir_relblock, text):
  try:
    cmd = f'rg -n -H -F --no-heading --hidden -g "!{{{out_dir_relblock},.git}}" -g "*.tex" "{text}"'
    out = subprocess.check_output(cmd, cwd=tex_root, shell=True, text=True, stderr=subprocess.DEVNULL)
    cands = []
    for line in out.splitlines()[:10]:
      m = re.match(r'^(.+\.tex):(\d+):(.*)$', line)
      if not m: continue
      file = m.group(1)
      if not os.path.isabs(file): file = os.path.join(tex_root, file)
      ln = int(m.group(2))
      cands.append({'method':'ripgrep','file':file,'line':ln,'snippet':read_context(file, ln)})
    return cands
  except Exception:
    return []

def score_candidate(cand, text):
  # 简单评分：越精确越高；synctex > rg
  score = 0.0
  score += 0.6 if cand['method']=='synctex' else 0.0
  # 行内容相似（忽略空白）
  line_txt = cand.get('snippet','')
  flat = re.sub(r'\s+', ' ', line_txt).lower()
  tar  = re.sub(r'\s+', ' ', text).lower()
  score += min(len(set(tar.split()) & set(flat.split()))/max(1,len(tar.split())), 0.3)
  return round(score, 3)

def main():
  payload = sys.stdin.read().strip()
  if not payload:
    print(json.dumps({'error':'no input'})); return
  data = json.loads(payload)
  selected = norm_text(data.get('selectedText',''))
  if not selected:
    print(json.dumps({'error':'no text'})); return

  tex_root = data['TEX_ROOT']; out_dir = data['OUT_DIR']; pdf = data['PDF_PATH']
  out_rel = os.path.relpath(out_dir, tex_root)

  # 1) 若前端给了坐标，先试 synctex（再得分）
  page, h, v = data.get('page'), data.get('h'), data.get('v')
  best = None
  if page and h is not None and v is not None and os.path.exists(pdf):
    best = synctex_query(pdf, page, h, v)
    if best: best['score'] = score_candidate(best, selected)

  # 2) ripgrep 候选
  cands = rg_candidates(tex_root, out_rel, selected)
  for c in cands:
    c['score'] = score_candidate(c, selected)

  # 汇总：若 synctex 有结果，把它放前面
  allc = ([best] if best else []) + cands
  allc = [x for x in allc if x]
  if not allc:
    print(json.dumps({'error':'not found'})); return
  # 如果只有一个，直接返回；否则返回 candidates
  if len(allc)==1:
    print(json.dumps(allc[0], ensure_ascii=False)); return
  # 多候选
  allc.sort(key=lambda x: x.get('score',0), reverse=True)
  print(json.dumps({'candidates': allc[:10]}, ensure_ascii=False))

if __name__ == '__main__':
  main()