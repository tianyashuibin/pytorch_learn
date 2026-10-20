from pathlib import Path
import re, html
root = Path(__file__).resolve().parent
assets = root / 'assets/session1'

def svg(name, title, body, height=230):
    text = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="img" aria-labelledby="title"><title id="title">{title}</title><rect width="960" height="{height}" rx="16" fill="#eff6ff"/><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#52718c"/></marker></defs><style>text{{font-family:Arial,'PingFang SC',sans-serif;fill:#18354d;font-size:20px}}.small{{font-size:17px}}.box{{fill:#fff;stroke:#bdd0df;stroke-width:2}}.line{{fill:none;stroke:#52718c;stroke-width:2;marker-end:url(#arrow)}}</style>{body}</svg>'''
    (assets/name).write_text(text, encoding='utf-8')

def flow(name,title,labels,sub):
    parts=[]
    n=len(labels); step=900/n; w=step-24
    for i,(label,detail) in enumerate(zip(labels,sub)):
        x=30+i*step
        parts += [f'<rect class="box" x="{x}" y="70" width="{w}" height="90" rx="12"/>', f'<text x="{x+w/2}" y="107" text-anchor="middle">{label}</text>', f'<text class="small" x="{x+w/2}" y="138" text-anchor="middle">{detail}</text>']
        if i<n-1: parts += [f'<path class="line" d="M{x+w+2} 115 H{x+step-4}"/>']
    svg(name,title,''.join(parts))
flow('learning-loop.svg','机器学习从数据到训练和评估的闭环',['数据与标签','特征与模型','损失与训练','评估与选择','新数据推理'],['可靠、可用','定义 fθ(x)','调整参数 θ','验证泛化','输出预测'])
flow('training-step.svg','一次训练迭代：清理梯度、前向、损失、反向和参数更新',['zero_grad','forward','loss','backward','step'],['清理已有梯度','计算预测','度量错误','计算梯度','更新参数'])
svg('recommender.svg','用户和物品 Embedding 与数值特征拼接后进入 MLP', '''<rect class="box" x="30" y="30" width="220" height="65" rx="10"/><text x="140" y="57" text-anchor="middle">用户 ID → Embedding</text><text class="small" x="140" y="81" text-anchor="middle">[B] → [B, 8]</text><rect class="box" x="30" y="120" width="220" height="65" rx="10"/><text x="140" y="147" text-anchor="middle">物品 ID → Embedding</text><text class="small" x="140" y="171" text-anchor="middle">[B] → [B, 8]</text><rect class="box" x="30" y="210" width="220" height="65" rx="10"/><text x="140" y="249" text-anchor="middle">数值特征 [B, 3]</text><path class="line" d="M250 63 L325 150"/><path class="line" d="M250 153 H325"/><path class="line" d="M250 243 L325 160"/><rect class="box" x="330" y="115" width="150" height="80" rx="10"/><text x="405" y="147" text-anchor="middle">特征拼接</text><text class="small" x="405" y="176" text-anchor="middle">[B, 19]</text><path class="line" d="M480 155 H530"/><rect class="box" x="535" y="115" width="180" height="80" rx="10"/><text x="625" y="147" text-anchor="middle">Linear + ReLU</text><text class="small" x="625" y="176" text-anchor="middle">[B, 16]</text><path class="line" d="M715 155 H765"/><rect class="box" x="770" y="115" width="160" height="80" rx="10"/><text x="850" y="147" text-anchor="middle">Linear → logit</text><text class="small" x="850" y="176" text-anchor="middle">[B] → sigmoid</text>''',310)
svg('gradient-step.svg','梯度更新后预测由2变为3.5，向目标5靠近', '''<text x="40" y="45">单样本：x = 2，y = 5；学习率 η = 0.1</text><path class="line" d="M80 135 H880"/><g text-anchor="middle"><circle cx="340" cy="135" r="10" fill="#d97706"/><text x="340" y="104">更新前：ŷ = 2</text><circle cx="550" cy="135" r="10" fill="#2563eb"/><text x="550" y="180">更新后：ŷ = 3.5</text><circle cx="760" cy="135" r="10" fill="#059669"/><text x="760" y="104">目标：y = 5</text></g><text class="small" x="40" y="215">w：1 → 1.6　 b：0 → 0.3　 损失：4.5 → 1.125</text>''',250)
svg('generalization.svg','概念示意：训练损失下降，验证损失先下降再上升', '''<path class="line" d="M95 35 V285 H890"/><text x="30" y="45">损失</text><text x="830" y="325">训练轮数</text><path d="M110 80 C200 175 320 220 460 245 S740 262 865 268" fill="none" stroke="#2563eb" stroke-width="4"/><path d="M110 65 C240 160 340 210 450 210 S670 130 865 85" fill="none" stroke="#d97706" stroke-width="4" stroke-dasharray="10 5"/><text x="680" y="248">训练损失（实线）</text><text x="650" y="65">验证损失（虚线）</text><path d="M450 215 V285" stroke="#8394a3" stroke-dasharray="5 5"/><text class="small" x="320" y="310">验证表现较好的区域</text><text class="small" x="95" y="365">概念示意图：不对应示例代码的实测结果。</text>''',390)

md=(root/'第一场_模型如何学习.md').read_text()
def inline(s):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s
lines=md.splitlines(); out=[]; nav=[]; i=0; section=0
while i<len(lines):
    s=lines[i]
    if not s.strip(): i+=1; continue
    if s.startswith('```'):
        language=s[3:]; block=[]; i+=1
        while i<len(lines) and not lines[i].startswith('```'): block.append(lines[i]); i+=1
        out.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>'); i+=1; continue
    if s.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            row=[v.strip() for v in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',v) for v in row): rows.append(row)
            i+=1
        out.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(v)+'</th>' for v in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(v)+'</td>' for v in row)+'</tr>' for row in rows[1:])+'</tbody></table></div>'); continue
    image=re.fullmatch(r'!\[([^\]]+)\]\(([^)]+)\)',s)
    if image:
        out.append(f'<figure><img src="{image[2]}" alt="{image[1]}"/><figcaption>{image[1]}</figcaption></figure>'); i+=1; continue
    h=re.match(r'^(#{1,3}) (.*)',s)
    if h:
        level=len(h[1]); section+=1; ident=f's{section}'
        out.append(f'<h{level} id="{ident}">{inline(h[2])}</h{level}>')
        if level==2: nav.append(f'<a href="#{ident}">{html.escape(h[2])}</a>')
        i+=1; continue
    if s.startswith('> '): out.append('<blockquote>'+inline(s[2:])+'</blockquote>'); i+=1; continue
    if re.match(r'^(- |\d+\. )',s):
        ordered=bool(re.match(r'^\d+\.',s)); tag='ol' if ordered else 'ul'; items=[]
        while i<len(lines) and re.match(r'^(- |\d+\. )',lines[i]):
            items.append('<li>'+inline(re.sub(r'^(- |\d+\. )','',lines[i]))+'</li>'); i+=1
        out.append(f'<{tag}>'+''.join(items)+f'</{tag}>'); continue
    out.append('<p>'+inline(s)+'</p>'); i+=1
content='\n'.join(out)
simulator='''<aside class="experiment"><h3>动手观察：一次梯度更新</h3><p>固定 x = 2、y = 5、w = 1、b = 0。拖动学习率，比较更新前后的预测与损失。每次都从相同初始参数计算一步。</p><label for="lr">学习率 η：<output id="lr-value">0.10</output></label><input id="lr" type="range" min="0" max="0.6" step="0.01" value="0.1"><div id="result" aria-live="polite"></div><div class="prediction-track"><span id="prediction">预测</span><span class="target">目标 y = 5</span></div><p class="small">本例的新损失为 4.5 × (1 − 5η)²。η = 0.2 时恰好拟合这一条样本；η &gt; 0.4 时，一步更新反而增大损失。这些阈值仅适用于当前示例。</p></aside>'''
content=content.replace('<h3 id="s', '<h3 id="s',1)
content=content.replace('<figure><img src="assets/session1/gradient-step.svg"',simulator+'<figure><img src="assets/session1/gradient-step.svg"')
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第一场：模型如何学习</title><style>
:root{color-scheme:light;--ink:#16324a;--muted:#516779;--accent:#176b83}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fa;color:var(--ink);font-family:system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;line-height:1.85}header{padding:60px max(24px,calc((100vw - 1160px)/2));background:#14364b;color:#fff}header p{color:#c5dae4;margin:8px 0}header h1{font-size:clamp(30px,4vw,48px);margin:12px 0;line-height:1.3}.layout{display:grid;grid-template-columns:220px minmax(0,900px);gap:32px;max-width:1200px;margin:36px auto;padding:0 24px}nav{position:sticky;top:24px;align-self:start;font-size:14px}nav a{display:block;text-decoration:none;color:var(--muted);padding:8px 0;border-bottom:1px solid #dce4eb}nav a:hover{color:var(--accent)}main{min-width:0;background:#fff;padding:36px 48px;border-radius:16px}main>h1{display:none}h2{font-size:28px;margin:56px 0 20px;border-top:1px solid #e1e8ef;padding-top:28px;scroll-margin-top:24px}h2:first-of-type{margin-top:0;border:0;padding:0}h3{font-size:21px;margin:32px 0 12px}p,li{font-size:17px}a{color:var(--accent)}blockquote{margin:0 0 28px;padding:12px 20px;border-left:4px solid var(--accent);background:#edf6f8}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:#eef3f7;padding:2px 5px;border-radius:4px;font-size:.9em}pre{background:#122838;color:#e1edf5;padding:24px;overflow:auto;border-radius:10px;line-height:1.65;font-size:14px}pre code{padding:0;background:none;color:inherit;font-size:inherit}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid #dce5ec}th{background:#edf4f8}figure{margin:30px 0}figure img{display:block;width:100%;height:auto}figcaption{font-size:14px;color:var(--muted);text-align:center;margin-top:10px}.experiment{background:#edf7f5;padding:24px;border-radius:12px}.experiment h3{margin-top:0}input[type=range]{display:block;width:100%;margin:16px 0;accent-color:var(--accent)}#result{font-variant-numeric:tabular-nums;font-weight:600;min-height:60px}.prediction-track{position:relative;margin:28px 0 40px;height:8px;background:#c8dce0;border-radius:8px}#prediction,.target{position:absolute;top:14px;transform:translateX(-50%);font-size:14px;white-space:nowrap}#prediction:before,.target:before{content:'';position:absolute;top:-18px;left:50%;width:12px;height:12px;border-radius:50%;background:#176b83}.target{left:62.5%;top:-32px}.target:before{top:28px;background:#b56c19}.small{font-size:14px;color:var(--muted)}footer{max-width:900px;margin:30px auto 60px;text-align:center;color:var(--muted);font-size:14px}.actions a,.actions button{display:inline-block;margin:12px 10px 0 0;padding:8px 14px;border:1px solid #6c95aa;border-radius:6px;background:transparent;color:#fff;font:inherit;text-decoration:none;cursor:pointer}@media(max-width:900px){.layout{display:block}nav{position:static;display:flex;flex-wrap:wrap;gap:8px 16px;margin-bottom:24px}nav a{border:0}main{padding:24px}header{padding:32px 24px}}@media(max-width:480px){.layout{padding:0 12px}main{padding:20px 16px}p,li{font-size:16px}h2{font-size:24px}th,td{min-width:110px}#prediction,.target{font-size:12px}}@media print{header{background:none;color:#000;padding:0}header p,header .actions,nav,.experiment{display:none}.layout{display:block;margin:0;padding:0}main{padding:0}h2{break-before:page}figure,table{break-inside:avoid}pre{white-space:pre-wrap;background:#f2f2f2;color:#000}footer{display:none}}
</style></head><body><header><p>算法模型分享 / SESSION 01 / 约 80 分钟</p><h1>模型如何学习</h1><p>从数据到参数，用 Embedding + MLP 走完一次训练闭环。</p><div class="actions"><a href="第一场_模型如何学习.md">阅读 Markdown 源讲义</a><button onclick="window.print()">打印讲义</button></div></header><div class="layout"><nav aria-label="讲义目录">NAV</nav><main>CONTENT</main></div><footer>第一场：模型如何学习 · 图示为教学示意 · 下一场：PyTorch 与 GPU 如何执行模型</footer><script>
const slider=document.getElementById('lr');function update(){const eta=Number(slider.value),w=1+6*eta,b=3*eta,p=2*w+b,L=.5*(p-5)**2;document.getElementById('lr-value').textContent=eta.toFixed(2);document.getElementById('result').textContent=`新参数 w = ${w.toFixed(2)}，b = ${b.toFixed(2)}；预测 = ${p.toFixed(2)}；损失 4.500 → ${L.toFixed(3)}（${L<4.5?'下降':L>4.5?'上升':'不变'}）`;const marker=document.getElementById('prediction');marker.style.left=`${p/8*100}%`;marker.textContent=`预测 ${p.toFixed(2)}`;}slider.addEventListener('input',update);update();
</script></body></html>'''
(root/'第一场_模型如何学习.html').write_text(page.replace('NAV',''.join(nav)).replace('CONTENT',content),encoding='utf-8')
print('Built Markdown-based HTML and 5 SVG illustrations.')
