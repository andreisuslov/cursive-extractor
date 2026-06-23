"""Local web tool to review + fix per-page word boxes, then re-vectorize.

Browse the output tree (documents -> pages); for each page see the image with its
detected word boxes and fix them by hand:

  * geometry  -- move/resize a box; EXPAND it when the word is cut; turn a box into
                 an arbitrary POLYGON (add/drag/delete vertices) to tightly outline
                 a slanted/irregular word; add a missing box; delete a junk box.
  * labels    -- an ordered box LIST with a text field per box (fix spelling), plus
                 SHIFT operations to re-align the whole word sequence to the boxes
                 (e.g. box 20 should hold box 21's word -> "pull labels up" from 20).
  * progress  -- mark a page done; the tree shows what's finished so you can stop
                 and resume later.
  * zoom/pan  -- slider, +/-, Fit, and trackpad pinch (ctrl-wheel); two-finger
                 scroll to pan.

Save writes the corrected boxes.json (with any polygon) back in place; "Save &
re-vectorize" also re-runs ocr.vectorize + ocr.package_boxes for the page.

    python -m ocr.box_editor                 # http://127.0.0.1:8765, root=outputs/
    python -m ocr.box_editor --root /path --port 8800

Stdlib only; single local user. box_2d is [ymin,xmin,ymax,xmax] on a 0-1000 scale
(page-relative); an optional polygon is a list of [x,y] in the same 0-1000 scale.
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


def _state_path() -> str:
    return os.path.join(ROOT, ".box_editor_state.json")


def _load_state() -> dict:
    p = _state_path()
    return _load_json(p) if os.path.exists(p) else {"done": {}}


def _save_state(st: dict) -> None:
    with open(_state_path(), "w") as f:
        json.dump(st, f, indent=2)


def _page_num(page_dir: str) -> int:
    return int(re.search(r"page_(\d+)", os.path.basename(page_dir)).group(1))


def _boxes_json(page_path: str) -> str | None:
    hits = glob.glob(os.path.join(page_path, "*_boxes.json"))
    return hits[0] if hits else None


def _pdf_for(doc: str) -> str | None:
    cand = os.path.join(ROOT, doc, f"{doc}.pdf")
    return cand if os.path.exists(cand) else None


def tree() -> list:
    st = _load_state()["done"]
    out = []
    for doc in sorted(os.listdir(ROOT)):
        ddir = os.path.join(ROOT, doc)
        if not os.path.isdir(ddir):
            continue
        pages = []
        for pdir in sorted(glob.glob(os.path.join(ddir, "page_*"))):
            bj = _boxes_json(pdir)
            if bj:
                pg = os.path.basename(pdir)
                pages.append(
                    {"page": pg, "n": len(_load_json(bj)), "done": st.get(f"{doc}/{pg}", False)}
                )
        if pages:
            out.append({"doc": doc, "pages": pages})
    return out


def page_image_bytes(doc: str, page: str) -> bytes | None:
    pdir = os.path.join(ROOT, doc, page)
    png = glob.glob(os.path.join(pdir, "*_page.png"))
    if png:
        return open(png[0], "rb").read()
    pdf = _pdf_for(doc)
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
        extra = ["--boxes", bj] if mod.endswith("vectorize") else ["--no-qa"]
        cmd = ["python3", "-m", mod, "--pdf", pdf, "--page", str(n), "--output-root", ROOT, *extra]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
        logs.append((r.stdout or "")[-300:] + (r.stderr or "")[-300:])
        if r.returncode != 0:
            return False, "\n".join(logs)
    return True, "\n".join(logs)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Box Editor</title>
<style>
:root{--bg:#15191e;--panel:#222831;--ink:#e6edf3;--mut:#9fb0c0;--line:#39424d;--box:#00aaa0;--sel:#ffcc00;--ok:#3fd07a}
*{box-sizing:border-box}body{margin:0;display:flex;height:100vh;font:13px system-ui,sans-serif;background:var(--bg);color:var(--ink)}
#side{width:210px;flex:0 0 210px;background:var(--panel);border-right:1px solid var(--line);overflow:auto;padding:8px}
#side h2{font-size:11px;color:var(--mut);margin:8px 4px}
.doc{font-weight:600;margin:8px 0 2px;font-size:12px}
.pg{padding:3px 8px;margin-left:6px;border-radius:5px;cursor:pointer;color:var(--mut)}
.pg:hover{background:#30373f}.pg.on{background:var(--box);color:#04201e}.pg.done{color:var(--ok)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:6px 10px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:7px;align-items:center;flex-wrap:wrap}
button{background:#30373f;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 9px;cursor:pointer;font-size:12px}
button:hover{background:#3a424c}button.on{background:var(--box);border-color:var(--box);color:#04201e;font-weight:600}
button.prim{background:var(--box);border-color:var(--box);color:#04201e;font-weight:600}
.sep{width:1px;height:22px;background:var(--line);margin:0 3px}
#cvwrap{flex:1;overflow:auto;background:#0e1217}
#cv{display:block}
#list{width:230px;flex:0 0 230px;background:var(--panel);border-left:1px solid var(--line);overflow:auto;padding:6px}
#list h2{font-size:11px;color:var(--mut);margin:6px 4px}
.row{display:flex;gap:5px;align-items:center;padding:2px 3px;border-radius:5px}
.row.on{background:#3a3320}.row .ix{width:30px;color:var(--mut);text-align:right;font-variant-numeric:tabular-nums;cursor:pointer}
.row input{flex:1;background:#30373f;border:1px solid var(--line);border-radius:5px;color:var(--ink);padding:3px 6px;min-width:0}
.pill{color:var(--mut);font-size:11px}
</style></head><body>
<div id="side"><h2>DOCUMENTS</h2><div id="tree"></div></div>
<div id="main">
 <div id="bar">
  <span id="title" class="pill">pick a page</span><span style="flex:1"></span>
  <button id="selBtn" class="on">Select</button>
  <button id="addBoxBtn">+ Box</button>
  <button id="addPtBtn">+ Point</button>
  <button id="delBtn">Del box (⌫)</button>
  <span class="sep"></span>
  <button id="zoutBtn">&minus;</button><input type="range" id="zoom" min="20" max="500" value="100" style="width:90px">
  <button id="zinBtn">+</button><button id="fitBtn">Fit</button><span id="zoomV" class="pill">100%</span>
  <span class="sep"></span>
  <button id="doneBtn">Mark done</button>
  <button id="saveBtn">Save</button><button id="reBtn" class="prim">Save &amp; re-vectorize</button>
  <span id="msg" class="pill"></span>
 </div>
 <div id="cvwrap"><canvas id="cv"></canvas></div>
</div>
<div id="list">
 <h2>WORDS (in order)</h2>
 <div class="pill" style="margin:2px 4px 6px">Select a box, then shift labels. Shift-click a 2nd row to bound the range.</div>
 <div style="display:flex;gap:5px;margin:0 2px 6px">
   <button id="pullBtn" title="each box takes the NEXT box's word">◀ Pull labels</button>
   <button id="pushBtn" title="each box takes the PREVIOUS box's word (blank inserted)">Push labels ▶</button>
 </div>
 <div id="rows"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d'),wrap=$('cvwrap');
let doc=null,page=null,IW=0,IH=0,scale=1,tool='select';
let boxes=[],sel=-1,rangeEnd=-1,drag=null,pageDone=false;

/* ---------- tree ---------- */
async function loadTree(){
  const t=await (await fetch('/api/tree')).json(); const el=$('tree'); el.innerHTML='';
  for(const d of t){ const dv=document.createElement('div');dv.className='doc';dv.textContent=d.doc;el.appendChild(dv);
    for(const p of d.pages){ const pv=document.createElement('div');pv.className='pg'+(p.done?' done':'');
      pv.dataset.k=d.doc+'|'+p.page; pv.textContent=(p.done?'✓ ':'')+p.page+'  ('+p.n+')';
      pv.onclick=()=>openPage(d.doc,p.page); el.appendChild(pv);} }
}
function markTreeActive(){document.querySelectorAll('.pg').forEach(x=>x.classList.toggle('on',x.dataset.k===doc+'|'+page));}

/* ---------- load page ---------- */
async function openPage(d,p){
  doc=d;page=p;sel=-1;rangeEnd=-1;
  const r=await (await fetch('/api/page?doc='+encodeURIComponent(d)+'&page='+encodeURIComponent(p))).json();
  pageDone=r.done; updateDoneBtn();
  const im=new Image(); im.onload=()=>{IW=im.naturalWidth;IH=im.naturalHeight;cv.width=IW;cv.height=IH;
    boxes=r.boxes.map(fromBox); img=im; fitZoom(); renderList(); markTreeActive();};
  im.src=r.img+'&t='+Date.now();
}
let img=null;
// boxes.json entry -> editor box (natural px). polygon -> shape 'poly', else 'rect'
function fromBox(b){
  if(b.polygon&&b.polygon.length>=3) return {text:b.text||'',shape:'poly',verts:b.polygon.map(([x,y])=>[x/1000*IW,y/1000*IH])};
  const[ymn,xmn,ymx,xmx]=b.box_2d; return {text:b.text||'',shape:'rect',x0:xmn/1000*IW,y0:ymn/1000*IH,x1:xmx/1000*IW,y1:ymx/1000*IH};
}
function toBox(o){ // editor box -> boxes.json entry
  let xs,ys;
  if(o.shape==='poly'){xs=o.verts.map(v=>v[0]);ys=o.verts.map(v=>v[1]);}
  else{xs=[o.x0,o.x1];ys=[o.y0,o.y1];}
  const X0=Math.min(...xs),Y0=Math.min(...ys),X1=Math.max(...xs),Y1=Math.max(...ys);
  const e={text:o.text,box_2d:[Math.round(Y0/IH*1000),Math.round(X0/IW*1000),Math.round(Y1/IH*1000),Math.round(X1/IW*1000)]};
  if(o.shape==='poly') e.polygon=o.verts.map(([x,y])=>[Math.round(x/IW*1000),Math.round(y/IH*1000)]);
  return e;
}

/* ---------- zoom / pan ---------- */
function setScale(z,ax,ay){
  z=Math.max(0.1,Math.min(8,z));
  let bx=0,by=0;
  if(ax!=null){const r=cv.getBoundingClientRect();bx=(ax-r.left)/r.width;by=(ay-r.top)/r.height;}
  scale=z; cv.style.width=(IW*scale)+'px'; cv.style.height=(IH*scale)+'px';
  $('zoom').value=Math.round(z*100); $('zoomV').textContent=Math.round(z*100)+'%'; draw();
  if(ax!=null){const r=cv.getBoundingClientRect(); wrap.scrollLeft+=(r.left+bx*r.width)-ax; wrap.scrollTop+=(r.top+by*r.height)-ay;}
}
function fitZoom(){ setScale(Math.min(1,(wrap.clientWidth-24)/IW)); }
$('zoom').oninput=()=>setScale($('zoom').value/100);
$('zinBtn').onclick=()=>setScale(scale*1.25); $('zoutBtn').onclick=()=>setScale(scale/1.25); $('fitBtn').onclick=fitZoom;
wrap.addEventListener('wheel',e=>{ if(!e.ctrlKey)return; e.preventDefault(); setScale(scale*(1-e.deltaY*0.01),e.clientX,e.clientY);},{passive:false});

/* ---------- draw ---------- */
function draw(){
  if(!img)return; ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0,IW,IH);
  const hs=Math.max(4,7/scale), lw=Math.max(1,1.6/scale);
  boxes.forEach((b,i)=>{const on=i===sel,inR=inRange(i);
    ctx.lineWidth=on?lw*1.8:lw; ctx.strokeStyle=on?'#ffcc00':(inR?'#ff8c42':'#00aaa0');
    const pts=polyOf(b); ctx.beginPath(); pts.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.closePath(); ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle; ctx.font=(13/scale)+'px sans-serif'; ctx.fillText((b.text||'')+' ['+i+']',pts[0][0],pts[0][1]-3/scale);
    if(on){const hh=handlesOf(b); ctx.fillStyle='#ffcc00'; hh.forEach(h=>ctx.fillRect(h[0]-hs/2,h[1]-hs/2,hs,hs));}
  });
}
function polyOf(b){ return b.shape==='poly'?b.verts:[[b.x0,b.y0],[b.x1,b.y0],[b.x1,b.y1],[b.x0,b.y1]]; }
function handlesOf(b){ // rect: 8 resize handles; poly: its vertices
  if(b.shape==='poly')return b.verts;
  const mx=(b.x0+b.x1)/2,my=(b.y0+b.y1)/2; return [[b.x0,b.y0],[mx,b.y0],[b.x1,b.y0],[b.x1,my],[b.x1,b.y1],[mx,b.y1],[b.x0,b.y1],[b.x0,my]];
}
const RH=['nw','n','ne','e','se','s','sw','w'];

/* ---------- hit-test + interaction ---------- */
function mouse(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];}
function inside(b,mx,my){const p=polyOf(b);let c=false;for(let i=0,j=p.length-1;i<p.length;j=i++){if((p[i][1]>my)!=(p[j][1]>my)&&mx<(p[j][0]-p[i][0])*(my-p[i][1])/(p[j][1]-p[i][1])+p[i][0])c=!c;}return c;}
cv.onmousedown=e=>{const[mx,my]=mouse(e),tol=8/scale;
  if(sel>=0){const hh=handlesOf(boxes[sel]);
    for(let i=0;i<hh.length;i++)if(Math.abs(mx-hh[i][0])<tol&&Math.abs(my-hh[i][1])<tol){
      drag=boxes[sel].shape==='poly'?{kind:'vert',i}:{kind:'resize',h:RH[i],b:{...boxes[sel]}};return;}}
  if(tool==='addpt'){for(let i=boxes.length-1;i>=0;i--)if(inside(boxes[i],mx,my)||near(boxes[i],mx,my,tol*2)){sel=i;addPoint(boxes[i],mx,my);renderList();draw();return;}return;}
  if(tool==='addbox'){boxes.push({text:'',shape:'rect',x0:mx,y0:my,x1:mx,y1:my});sel=boxes.length-1;drag={kind:'new'};renderList();return;}
  for(let i=boxes.length-1;i>=0;i--)if(inside(boxes[i],mx,my)){selectBox(i,e.shiftKey);drag={kind:'move',ox:mx,oy:my,snap:JSON.parse(JSON.stringify(boxes[i]))};return;}
  sel=-1;rangeEnd=-1;renderList();draw();
};
cv.onmousemove=e=>{if(!drag)return;const[mx,my]=mouse(e);const b=boxes[sel];
  if(drag.kind==='move'){const dx=mx-drag.ox,dy=my-drag.oy;const s=drag.snap;
    if(b.shape==='poly')b.verts=s.verts.map(([x,y])=>[x+dx,y+dy]);else{b.x0=s.x0+dx;b.y0=s.y0+dy;b.x1=s.x1+dx;b.y1=s.y1+dy;}}
  else if(drag.kind==='vert'){b.verts[drag.i]=[mx,my];}
  else if(drag.kind==='new'){b.x1=mx;b.y1=my;}
  else if(drag.kind==='resize'){const h=drag.h,o=drag.b;b.x0=o.x0;b.y0=o.y0;b.x1=o.x1;b.y1=o.y1;
    if(h.includes('n'))b.y0=my;if(h.includes('s'))b.y1=my;if(h.includes('w'))b.x0=mx;if(h.includes('e'))b.x1=mx;}
  draw();
};
window.onmouseup=()=>{if(drag&&drag.kind==='new'){const b=boxes[sel];if(Math.abs(b.x1-b.x0)<3){boxes.pop();sel=-1;}else{b.x0=Math.min(b.x0,b.x1);/*normalize on save*/}}drag=null;};
cv.ondblclick=e=>{const[mx,my]=mouse(e),tol=9/scale; // delete a polygon vertex
  if(sel>=0&&boxes[sel].shape==='poly'&&boxes[sel].verts.length>3){const v=boxes[sel].verts;
    for(let i=0;i<v.length;i++)if(Math.abs(mx-v[i][0])<tol&&Math.abs(my-v[i][1])<tol){v.splice(i,1);draw();return;}}
};
function near(b,mx,my,tol){const p=polyOf(b);for(let i=0,j=p.length-1;i<p.length;j=i++)if(distSeg(mx,my,p[j],p[i])<tol)return true;return false;}
function distSeg(px,py,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],l=dx*dx+dy*dy;let t=l?((px-a[0])*dx+(py-a[1])*dy)/l:0;t=Math.max(0,Math.min(1,t));return Math.hypot(px-a[0]-t*dx,py-a[1]-t*dy);}
function addPoint(b,mx,my){ // promote rect->poly if needed, then insert vertex on nearest edge
  if(b.shape!=='poly'){b.shape='poly';b.verts=[[b.x0,b.y0],[b.x1,b.y0],[b.x1,b.y1],[b.x0,b.y1]];delete b.x0;delete b.y0;delete b.x1;delete b.y1;}
  const v=b.verts;let best=0,bd=1e9;for(let i=0;i<v.length;i++){const d=distSeg(mx,my,v[i],v[(i+1)%v.length]);if(d<bd){bd=d;best=i;}}
  v.splice(best+1,0,[mx,my]);
}
function inRange(i){if(sel<0||rangeEnd<0)return false;const a=Math.min(sel,rangeEnd),b=Math.max(sel,rangeEnd);return i>=a&&i<=b&&i!==sel;}
function selectBox(i,shift){if(shift&&sel>=0)rangeEnd=i;else{sel=i;rangeEnd=-1;}renderList();draw();
  const row=$('rows').children[i]; if(row)row.scrollIntoView({block:'nearest'});}

/* ---------- tools ---------- */
function setTool(t){tool=t;for(const[id,v]of[['selBtn','select'],['addBoxBtn','addbox'],['addPtBtn','addpt']])$(id).classList.toggle('on',v===t);}
$('selBtn').onclick=()=>setTool('select');$('addBoxBtn').onclick=()=>setTool('addbox');$('addPtBtn').onclick=()=>setTool('addpt');
$('delBtn').onclick=()=>{if(sel>=0){boxes.splice(sel,1);sel=-1;rangeEnd=-1;renderList();draw();}};
document.addEventListener('keydown',e=>{if((e.key==='Backspace'||e.key==='Delete')&&e.target.tagName!=='INPUT'&&sel>=0){e.preventDefault();$('delBtn').onclick();}});

/* ---------- ordered word list + label shifting ---------- */
function renderList(){const el=$('rows');el.innerHTML='';
  boxes.forEach((b,i)=>{const row=document.createElement('div');row.className='row'+(i===sel?' on':'');
    const ix=document.createElement('span');ix.className='ix';ix.textContent=i;ix.onclick=e=>{selectBox(i,e.shiftKey);};
    const inp=document.createElement('input');inp.value=b.text||'';inp.oninput=()=>{b.text=inp.value;draw();};
    inp.onfocus=()=>{sel=i;rangeEnd=-1;document.querySelectorAll('.row').forEach((r,k)=>r.classList.toggle('on',k===i));draw();};
    row.appendChild(ix);row.appendChild(inp);el.appendChild(row);});
}
function shiftLabels(dir){if(sel<0)return;let a=sel,b=rangeEnd>=0?rangeEnd:boxes.length-1;if(b<a)[a,b]=[b,a];
  if(dir<0){for(let i=a;i<b;i++)boxes[i].text=boxes[i+1].text;boxes[b].text='';}
  else{for(let i=b;i>a;i--)boxes[i].text=boxes[i-1].text;boxes[a].text='';}
  renderList();draw();}
$('pullBtn').onclick=()=>shiftLabels(-1);$('pushBtn').onclick=()=>shiftLabels(1);

/* ---------- done / save ---------- */
function updateDoneBtn(){$('doneBtn').textContent=pageDone?'✓ Done (undo)':'Mark done';$('doneBtn').classList.toggle('on',pageDone);}
$('doneBtn').onclick=async()=>{pageDone=!pageDone;updateDoneBtn();
  await fetch('/api/done',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({doc,page,done:pageDone})});loadTree();};
async function save(re){$('msg').textContent=re?'re-vectorizing…':'saving…';
  const r=await(await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({doc,page,boxes:boxes.map(toBox),reprocess:!!re})})).json();
  $('msg').textContent=r.ok?(re?'✓ saved + re-vectorized':'✓ saved'):('✗ '+(r.error||'error'));}
$('saveBtn').onclick=()=>save(false);$('reBtn').onclick=()=>save(true);
window.addEventListener('resize',()=>{if(IW)draw();});
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
            done = _load_state()["done"].get(f"{doc}/{page}", False)
            img = f"/img?doc={urllib.parse.quote(doc)}&page={urllib.parse.quote(page)}"
            return self._send(200, json.dumps({"boxes": boxes, "img": img, "done": done}))
        if u.path == "/img":
            data = page_image_bytes(q["doc"][0], q["page"][0])
            return self._send(200 if data else 404, data or b"", "image/png")
        return self._send(404, b"", "text/plain")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if path == "/api/done":
            st = _load_state()
            key = f"{body['doc']}/{body['page']}"
            if body["done"]:
                st["done"][key] = True
            else:
                st["done"].pop(key, None)
            _save_state(st)
            return self._send(200, json.dumps({"ok": True}))
        if path == "/api/save":
            doc, page, boxes = body["doc"], body["page"], body["boxes"]
            bj = _boxes_json(os.path.join(ROOT, doc, page))
            if not bj:
                sys.path.insert(0, REPO)
                from ocr import paths

                pdf = _pdf_for(doc)
                bj = os.path.join(
                    ROOT, doc, page, f"{paths.prefix(pdf, _page_num(page))}_boxes.json"
                )
            clean = []
            for b in boxes:
                e = {"text": b.get("text", ""), "box_2d": b["box_2d"]}
                if b.get("polygon"):
                    e["polygon"] = b["polygon"]
                clean.append(e)
            with open(bj, "w") as f:
                json.dump(clean, f, indent=2)
            if not body.get("reprocess"):
                return self._send(200, json.dumps({"ok": True}))
            ok, log = reprocess(doc, page)
            return self._send(200, json.dumps({"ok": ok, "error": None if ok else log[-300:]}))
        return self._send(404, json.dumps({"ok": False}))


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
