"""Local web tool to review + fix per-page word boxes, then re-vectorize.

Browse the output tree (documents -> pages), see each page image with its detected
word boxes overlaid, then fix them by hand: move/resize a box, EXPAND it when the
word is cut, delete junk boxes, add a missing one, edit the text. Save writes the
corrected boxes.json back in place; "Save & re-vectorize" also re-runs
``ocr.vectorize`` + ``ocr.package_boxes`` for that page so the strokes/crops are
regenerated from the corrected boxes.

    python -m ocr.box_editor                 # serves http://127.0.0.1:8765, root=outputs/
    python -m ocr.box_editor --root /path --port 8800

Stdlib only; single local user. box_2d is [ymin,xmin,ymax,xmax] on a 0-1000 scale
(page-relative), so edits are resolution-independent and feed straight back into
the pipeline.
"""

import argparse
import contextlib
import glob
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "outputs"


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _page_num(page_dir: str) -> int:
    return int(re.search(r"page_(\d+)", os.path.basename(page_dir)).group(1))


def _boxes_json(page_path: str) -> str | None:
    hits = glob.glob(os.path.join(page_path, "*_boxes.json"))
    return hits[0] if hits else None


def _pdf_for(doc: str) -> str | None:
    cand = os.path.join(ROOT, doc, f"{doc}.pdf")
    return cand if os.path.exists(cand) else None


def tree() -> list:
    out = []
    for doc in sorted(os.listdir(ROOT)):
        ddir = os.path.join(ROOT, doc)
        if not os.path.isdir(ddir):
            continue
        pages = []
        for pdir in sorted(glob.glob(os.path.join(ddir, "page_*"))):
            bj = _boxes_json(pdir)
            if bj:
                pages.append({"page": os.path.basename(pdir), "n": len(_load_json(bj))})
        if pages:
            out.append({"doc": doc, "pages": pages})
    return out


def page_image_bytes(doc: str, page: str) -> bytes | None:
    pdir = os.path.join(ROOT, doc, page)
    png = glob.glob(os.path.join(pdir, "*_page.png"))
    if png:
        return open(png[0], "rb").read()
    pdf = _pdf_for(doc)  # fall back to rendering the PDF page
    if not pdf:
        return None
    sys.path.insert(0, REPO)
    from ocr.pdf_utils import load_page

    img = load_page(pdf, _page_num(pdir) - 1, dpi=200).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def reprocess(doc: str, page: str) -> tuple[bool, str]:
    pdir = os.path.join(ROOT, doc, page)
    pdf, bj, n = _pdf_for(doc), _boxes_json(pdir), _page_num(pdir)
    if not (pdf and bj):
        return False, "missing PDF or boxes.json"
    logs = []
    for mod in ("ocr.vectorize", "ocr.package_boxes"):
        cmd = [
            "python3",
            "-m",
            mod,
            "--pdf",
            pdf,
            "--page",
            str(n),
            "--output-root",
            ROOT,
            *(["--boxes", bj] if mod.endswith("vectorize") else ["--no-qa"]),
        ]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        logs.append((r.stdout or "")[-400:] + (r.stderr or "")[-400:])
        if r.returncode != 0:
            return False, "\n".join(logs)
    return True, "\n".join(logs)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Box Editor</title>
<style>
:root{--bg:#15191e;--panel:#222831;--ink:#e6edf3;--mut:#9fb0c0;--line:#39424d;--sel:#ffcc00;--box:#00aaa0}
*{box-sizing:border-box}body{margin:0;display:flex;height:100vh;font:13px system-ui,sans-serif;background:var(--bg);color:var(--ink)}
#side{width:240px;flex:0 0 240px;background:var(--panel);border-right:1px solid var(--line);overflow:auto;padding:8px}
#side h2{font-size:12px;color:var(--mut);margin:8px 4px}
.doc{font-weight:600;margin:6px 0 2px;cursor:default}
.pg{padding:3px 8px;margin-left:8px;border-radius:5px;cursor:pointer;color:var(--mut)}
.pg:hover{background:#30373f}.pg.on{background:var(--box);color:#04201e}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:8px 10px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:#30373f;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}
button:hover{background:#3a424c}button.prim{background:var(--box);border-color:var(--box);color:#04201e;font-weight:600}
#cvwrap{flex:1;overflow:auto;padding:14px;background:#0e1217}
#cv{display:block;background:#fff;box-shadow:0 0 0 1px var(--line)}
#txt{padding:4px 8px;background:#30373f;border:1px solid var(--line);border-radius:6px;color:var(--ink);width:160px}
.pill{color:var(--mut);font-size:12px}
</style></head><body>
<div id="side"><h2>DOCUMENTS</h2><div id="tree"></div></div>
<div id="main">
 <div id="bar">
   <span id="title" class="pill">pick a page</span>
   <span style="flex:1"></span>
   <button id="addBtn">+ Add box</button>
   <button id="delBtn">Delete (⌫)</button>
   <span class="pill">Word:</span><input id="txt" placeholder="selected word text (edit to fix spelling)"/>
   <button id="saveBtn">Save</button>
   <button id="reBtn" class="prim">Save &amp; re-vectorize</button>
   <span id="msg" class="pill"></span>
 </div>
 <div id="cvwrap"><canvas id="cv"></canvas></div>
</div>
<script>
const $=id=>document.getElementById(id);
let doc=null,page=null,IW=0,IH=0,scale=1,img=new Image(),boxes=[],sel=-1;
let mode=null,drag=null; // drag={kind,ox,oy, handle}
const cv=$('cv'),ctx=cv.getContext('2d');

async function loadTree(){
  const t=await (await fetch('/api/tree')).json();
  const el=$('tree'); el.innerHTML='';
  for(const d of t){
    const dv=document.createElement('div'); dv.className='doc'; dv.textContent=d.doc; el.appendChild(dv);
    for(const p of d.pages){
      const pv=document.createElement('div'); pv.className='pg'; pv.textContent=p.page+'  ('+p.n+')';
      pv.onclick=()=>{document.querySelectorAll('.pg').forEach(x=>x.classList.remove('on'));pv.classList.add('on');open(d.doc,p.page);};
      el.appendChild(pv);
    }
  }
}
async function open(d,p){
  doc=d;page=p;sel=-1;
  const r=await (await fetch('/api/page?doc='+encodeURIComponent(d)+'&page='+encodeURIComponent(p))).json();
  boxes=r.boxes; $('title').textContent=d+' / '+p+' — '+boxes.length+' boxes';
  img=new Image(); img.onload=()=>{IW=img.naturalWidth;IH=img.naturalHeight;fit();draw();}; img.src=r.img+'&t='+Date.now();
}
function fit(){ const wrap=$('cvwrap'); scale=Math.min(1,(wrap.clientWidth-30)/IW); cv.width=IW*scale; cv.height=IH*scale; }
// box_2d [ymin,xmin,ymax,xmax] 0-1000  ->  canvas px
function toPx(b){const[ymn,xmn,ymx,xmx]=b.box_2d;return[xmn/1000*cv.width,ymn/1000*cv.height,xmx/1000*cv.width,ymx/1000*cv.height];}
function setPx(b,x0,y0,x1,y1){b.box_2d=[Math.round(Math.min(y0,y1)/cv.height*1000),Math.round(Math.min(x0,x1)/cv.width*1000),Math.round(Math.max(y0,y1)/cv.height*1000),Math.round(Math.max(x0,x1)/cv.width*1000)];}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height); if(img.complete)ctx.drawImage(img,0,0,cv.width,cv.height);
  boxes.forEach((b,i)=>{const[x0,y0,x1,y1]=toPx(b);
    ctx.lineWidth=i===sel?2.5:1.5; ctx.strokeStyle=i===sel?'#ffcc00':'#00aaa0';
    ctx.strokeRect(x0,y0,x1-x0,y1-y0);
    ctx.fillStyle=ctx.strokeStyle; ctx.font='12px sans-serif'; ctx.fillText(b.text||'',x0,Math.max(10,y0-3));
    if(i===sel)for(const[hx,hy]of handles(x0,y0,x1,y1)){ctx.fillStyle='#ffcc00';ctx.fillRect(hx-4,hy-4,8,8);}
  });
}
function handles(x0,y0,x1,y1){const mx=(x0+x1)/2,my=(y0+y1)/2;return[[x0,y0],[mx,y0],[x1,y0],[x1,my],[x1,y1],[mx,y1],[x0,y1],[x0,my]];}
const HN=['nw','n','ne','e','se','s','sw','w'];
function mouse(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left),(e.clientY-r.top)];}
cv.onmousedown=e=>{const[mx,my]=mouse(e);
  if(sel>=0){const[x0,y0,x1,y1]=toPx(boxes[sel]);const hs=handles(x0,y0,x1,y1);
    for(let i=0;i<hs.length;i++)if(Math.abs(mx-hs[i][0])<6&&Math.abs(my-hs[i][1])<6){drag={kind:'resize',handle:HN[i],x0,y0,x1,y1};return;}}
  // select topmost box under cursor
  for(let i=boxes.length-1;i>=0;i--){const[x0,y0,x1,y1]=toPx(boxes[i]);if(mx>=x0&&mx<=x1&&my>=y0&&my<=y1){sel=i;$('txt').value=boxes[i].text||'';drag={kind:'move',ox:mx,oy:my,b:[x0,y0,x1,y1]};draw();return;}}
  if(mode==='add'){boxes.push({text:'',box_2d:[0,0,0,0]});sel=boxes.length-1;drag={kind:'new',ox:mx,oy:my};return;}
  sel=-1;$('txt').value='';draw();
};
cv.onmousemove=e=>{if(!drag)return;const[mx,my]=mouse(e);const b=boxes[sel];
  if(drag.kind==='move'){const dx=mx-drag.ox,dy=my-drag.oy;setPx(b,drag.b[0]+dx,drag.b[1]+dy,drag.b[2]+dx,drag.b[3]+dy);}
  else if(drag.kind==='new'){setPx(b,drag.ox,drag.oy,mx,my);}
  else if(drag.kind==='resize'){let{x0,y0,x1,y1,handle:h}=drag;if(h.includes('n'))y0=my;if(h.includes('s'))y1=my;if(h.includes('w'))x0=mx;if(h.includes('e'))x1=mx;setPx(b,x0,y0,x1,y1);}
  draw();
};
window.onmouseup=()=>{if(drag&&drag.kind==='move')drag.b=null;drag=null;if(mode==='add'){mode=null;$('addBtn').classList.remove('prim');}};
$('addBtn').onclick=()=>{mode=mode==='add'?null:'add';$('addBtn').classList.toggle('prim',mode==='add');};
$('delBtn').onclick=()=>{if(sel>=0){boxes.splice(sel,1);sel=-1;$('txt').value='';draw();}};
document.addEventListener('keydown',e=>{if((e.key==='Backspace'||e.key==='Delete')&&document.activeElement!==$('txt')&&sel>=0){e.preventDefault();$('delBtn').onclick();}});
$('txt').oninput=()=>{if(sel>=0){boxes[sel].text=$('txt').value;draw();}};
async function save(re){
  $('msg').textContent=re?'re-vectorizing…':'saving…';
  const r=await (await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({doc,page,boxes,reprocess:!!re})})).json();
  $('msg').textContent=r.ok?(re?'✓ saved + re-vectorized':'✓ saved'):('✗ '+(r.error||'error'));
  if(re&&r.ok)open(doc,page);
}
$('saveBtn').onclick=()=>save(false); $('reBtn').onclick=()=>save(true);
window.addEventListener('resize',()=>{if(IW){fit();draw();}});
loadTree();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/api/tree":
            return self._send(200, json.dumps(tree()))
        if u.path == "/api/page":
            doc, page = q["doc"][0], q["page"][0]
            bj = _boxes_json(os.path.join(ROOT, doc, page))
            boxes = _load_json(bj) if bj else []
            img = f"/img?doc={urllib.parse.quote(doc)}&page={urllib.parse.quote(page)}"
            return self._send(200, json.dumps({"boxes": boxes, "img": img}))
        if u.path == "/img":
            data = page_image_bytes(q["doc"][0], q["page"][0])
            return self._send(200 if data else 404, data or b"", "image/png")
        return self._send(404, b"", "text/plain")

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/save":
            return self._send(404, json.dumps({"ok": False}))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        doc, page, boxes = body["doc"], body["page"], body["boxes"]
        bj = _boxes_json(os.path.join(ROOT, doc, page))
        if not bj:  # a page with no boxes.json yet -> create the canonical name
            sys.path.insert(0, REPO)
            from ocr import paths

            pdf = _pdf_for(doc)
            bj = os.path.join(ROOT, doc, page, f"{paths.prefix(pdf, _page_num(page))}_boxes.json")
        clean = [{"text": b.get("text", ""), "box_2d": b["box_2d"]} for b in boxes]
        with open(bj, "w") as f:
            json.dump(clean, f, indent=2)
        if not body.get("reprocess"):
            return self._send(200, json.dumps({"ok": True}))
        ok, log = reprocess(doc, page)
        return self._send(200, json.dumps({"ok": ok, "error": None if ok else log[-300:]}))


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join(REPO, "outputs"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    ROOT = args.root
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Box editor: http://127.0.0.1:{args.port}   (root: {ROOT})\nCtrl-C to stop.")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()


if __name__ == "__main__":
    main()
