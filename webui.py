import io, os, threading, time, uuid, json
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file, abort
from lumina_vec.pipeline import PRESETS, auto_params, vectorize
import cairosvg

ROOT=Path('/data'); UP=ROOT/'uploads'; OUT=ROOT/'outputs'; WEIGHTS=Path('/app/weights/realesr-animevideov3.pth')
UP.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=100*1024*1024
jobs={}; lock=threading.Lock()

@app.get('/')
def index(): return Path('/app/index.html').read_text(encoding='utf8')

def work(job, raw, preset):
    try:
        im=Image.open(io.BytesIO(raw)); im.load(); w,h=im.size
        if w*h>40_000_000: raise ValueError('图片超过4000万像素上限')
        name,grain,params=auto_params(im) if preset=='auto' else (preset,0,dict(PRESETS[preset]))
        started=time.monotonic(); svg,_=vectorize(im, str(WEIGHTS), ai=True, **params)
        svgpath=OUT/f'{job}.svg'; pngpath=OUT/f'{job}.png'; svgpath.write_text(svg,encoding='utf8')
        cairosvg.svg2png(bytestring=svg.encode(),write_to=str(pngpath),output_width=w,output_height=h)
        with lock: jobs[job]={'status':'done','svg':f'/file/{job}.svg','png':f'/file/{job}.png','width':w,'height':h,'paths':svg.count('<path'),'bytes':svgpath.stat().st_size,'seconds':round(time.monotonic()-started,2),'preset':name,'grain':round(grain,2),'params':params}
    except Exception as e:
        with lock: jobs[job]={'status':'error','error':str(e)[:500]}

@app.post('/api/process')
def process():
    up=request.files.get('image'); preset=request.form.get('preset','auto')
    if not up or not up.filename: return jsonify(error='请选择图片'),400
    if preset not in ('auto',*PRESETS): return jsonify(error='预设无效'),400
    raw=up.read(); job=uuid.uuid4().hex
    (UP/f'{job}.source').write_bytes(raw)
    with lock: jobs[job]={'status':'queued'}
    threading.Thread(target=work,args=(job,raw,preset),daemon=True).start()
    return jsonify(job=job,status='queued')

@app.get('/api/status/<job>')
def status(job):
    with lock: result=jobs.get(job)
    return (jsonify(result),200) if result else (jsonify(error='任务不存在'),404)

@app.get('/file/<name>')
def file(name):
    if Path(name).name!=name or Path(name).suffix not in ('.svg','.png'): abort(404)
    path=OUT/name
    if not path.is_file(): abort(404)
    return send_file(path,as_attachment=request.args.get('download')=='1',download_name=name)

@app.get('/healthz')
def health(): return jsonify(status='ok',weights=WEIGHTS.is_file(),presets=['auto',*PRESETS])
