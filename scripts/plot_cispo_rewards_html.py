#!/usr/bin/env python3
"""Generate a standalone interactive HTML reward plot from a CISPO receipt."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    history = receipt["history"]
    rollouts = int(receipt["collection"]["rollouts_per_prompt"])
    rewards = [sum(row["correct_counts"]) / (len(row["correct_counts"]) * rollouts)
               for row in history]
    groups = int(receipt["collection"].get("seed_groups", 0))
    total = int(receipt["collection"]["total_rollouts"])
    title = f"CISPO reward - {rollouts} rollouts per seed"
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>
<style>:root{{--p:#f3f0e7;--i:#171712;--m:#67665d;--o:#788600;--a:#dfff42;--g:#cbc8ba}}*{{box-sizing:border-box}}body{{margin:0;background:var(--p);color:var(--i);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}main{{padding:34px 42px}}h1{{font:800 clamp(28px,4vw,48px)/1 Arial,sans-serif;letter-spacing:-.045em;margin:8px 0 24px}}.chart{{border:1px solid var(--g);padding:18px;background:#ffffff24}}svg{{width:100%;height:auto;display:block}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--g);border-top:0}}.stats div{{padding:14px;border-right:1px solid var(--g)}}.stats div:last-child{{border:0}}b{{display:block;font:700 24px Arial,sans-serif}}span,.note{{font-size:11px;color:var(--m)}}.legend{{display:flex;gap:22px;font-size:12px;margin-top:12px}}.line{{display:inline-block;width:28px;height:5px;background:var(--o);margin-right:7px}}.mean{{background:var(--i);height:7px}}.tip{{position:fixed;display:none;background:var(--i);color:var(--p);padding:8px;font-size:12px;pointer-events:none}}@media(max-width:700px){{main{{padding:20px 14px}}.stats{{grid-template-columns:repeat(2,1fr)}}}}</style></head>
<body><main><span>BANKING77 · GPT-OSS-20B · FIXED QUALIFYING SEEDS</span><h1>{html.escape(title)}</h1><div class='chart'><svg id='p' viewBox='0 0 1100 540'></svg><div class='legend'><div><i class='line'></i>batch mean reward</div><div><i class='line mean'></i>cumulative mean</div></div><p class='note'>Rewards come from 64 fresh CISPO rollouts on the immutable qualifying-seed set. Hover over a point for exact values.</p></div><div class='stats'><div><b>{groups}</b><span>FIXED SEEDS</span></div><div><b>{total:,}</b><span>TRAINING ROLLOUTS</span></div><div><b>{len(history)}</b><span>OPTIMIZER UPDATES</span></div><div><b>{sum(len(x['correct_counts']) for x in history)}</b><span>GROUP PRESENTATIONS</span></div></div></main><div class='tip' id='t'></div>
<script>const r={json.dumps(rewards)},c=r.map((_,i)=>r.slice(0,i+1).reduce((a,b)=>a+b,0)/(i+1)),s=document.querySelector('#p'),N='http://www.w3.org/2000/svg',W=1100,H=540,m={{l:70,r:25,t:25,b:55}},mx=Math.max(.4,...r)*1.08,x=i=>m.l+i/(r.length-1)*(W-m.l-m.r),y=v=>H-m.b-v/mx*(H-m.t-m.b),e=(n,a={{}})=>{{let q=document.createElementNS(N,n);Object.entries(a).forEach(([k,v])=>q.setAttribute(k,v));return q}};for(let k=0;k<=5;k++){{let v=mx*k/5,Y=y(v);s.append(e('line',{{x1:m.l,y1:Y,x2:W-m.r,y2:Y,stroke:'#cbc8ba'}}));let z=e('text',{{x:m.l-12,y:Y+5,'text-anchor':'end',fill:'#67665d','font-size':13}});z.textContent=v.toFixed(2);s.append(z)}}let path=v=>v.map((q,i)=>(i?'L':'M')+x(i)+','+y(q)).join(' ');s.append(e('path',{{d:path(r),fill:'none',stroke:'#788600','stroke-width':4}}));s.append(e('path',{{d:path(c),fill:'none',stroke:'#171712','stroke-width':7}}));let t=document.querySelector('#t');r.forEach((v,i)=>{{let q=e('circle',{{cx:x(i),cy:y(v),r:5,fill:'#788600'}});q.onmouseenter=a=>{{t.style.display='block';t.style.left=a.clientX+12+'px';t.style.top=a.clientY+12+'px';t.textContent=`update ${{i+1}} · reward ${{v.toFixed(4)}} · cumulative ${{c[i].toFixed(4)}}`}};q.onmouseleave=()=>t.style.display='none';s.append(q)}});[1,10,20,30,40,50].forEach(v=>{{let z=e('text',{{x:x(v-1),y:H-18,'text-anchor':'middle',fill:'#67665d','font-size':13}});z.textContent=v;s.append(z)}})</script></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
