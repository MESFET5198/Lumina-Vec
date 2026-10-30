import io, os, re, threading, time, uuid
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file, abort
from lumina_vec.pipeline import PRESETS, auto_params, grain_level, vectorize
import cairosvg

ROOT = Path(os.environ.get('LVEC_DATA', '/data')); UP = ROOT / 'uploads'; OUT = ROOT / 'outputs'
WEIGHTS = Path(os.environ.get('LVEC_WEIGHTS', '/app/weights/realesr-animevideov3.pth'))
INDEX = Path(__file__).with_name('index.html')
UP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__); app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

jobs = OrderedDict()            # job id -> state dict (newest last)
lock = threading.Lock()
runner = threading.Semaphore(1)  # CPU-heavy: run one conversion at a time, queue the rest
MAX_PIXELS = 40_000_000
MAX_JOBS_KEPT = 50


def _set(job, **kw):
    with lock:
        jobs[job].update(kw)


def _queue_position(job):
    with lock:
        waiting = [j for j, s in jobs.items() if s['status'] == 'queued']
    return waiting.index(job) if job in waiting else 0


def work(job, raw, preset, overrides):
    try:
        im = Image.open(io.BytesIO(raw)); im.load(); w, h = im.size
        if w * h > MAX_PIXELS:
            raise ValueError(f'图片超过 {MAX_PIXELS // 1_000_000} MP 上限')
        with runner:
            _set(job, status='running', started=time.time())
            if preset == 'auto':
                name, grain, params = auto_params(im)
            else:
                name, grain, params = preset, grain_level(im), dict(PRESETS[preset])
            params.update(overrides)
            t = time.monotonic()
            svg, _ = vectorize(im, str(WEIGHTS), ai=True, **params)
            secs = time.monotonic() - t
            svgpath = OUT / f'{job}.svg'; pngpath = OUT / f'{job}.png'
            svgpath.write_text(svg, encoding='utf8')
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(pngpath), output_width=w, output_height=h)
            colors = len(set(m.lower() for m in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)))
            _set(job, status='done', svg=f'/file/{job}.svg', png=f'/file/{job}.png', width=w, height=h,
                 paths=svg.count('<path'), colors=colors, bytes=svgpath.stat().st_size,
                 seconds=round(secs, 1), preset=name, grain=round(grain, 2), params=params, finished=time.time())
    except Exception as e:
        _set(job, status='error', error=str(e)[:500])


@app.get('/')
def index():
    return INDEX.read_text(encoding='utf8')


@app.post('/api/process')
def process():
    up = request.files.get('image'); preset = request.form.get('preset', 'auto')
    if not up or not up.filename:
        return jsonify(error='请选择图片'), 400
    if preset not in ('auto', *PRESETS):
        return jsonify(error='预设无效'), 400
    overrides = {}
    bands = request.form.get('bands', '').strip()
    if bands:
        if not bands.isdigit() or not (8 <= int(bands) <= 256):
            return jsonify(error='颜色数需在 8–256 之间'), 400
        overrides['bands'] = int(bands)
    ink = request.form.get('ink', '').strip()
    if ink in ('0', '1'):
        overrides['ink'] = ink == '1'
    raw = up.read(); job = uuid.uuid4().hex
    (UP / f'{job}.source').write_bytes(raw)
    with lock:
        jobs[job] = {'status': 'queued', 'name': Path(up.filename).name[:120], 'created': time.time()}
        while len(jobs) > MAX_JOBS_KEPT:
            jobs.popitem(last=False)
    threading.Thread(target=work, args=(job, raw, preset, overrides), daemon=True).start()
    return jsonify(job=job, status='queued')


@app.get('/api/status/<job>')
def status(job):
    with lock:
        result = dict(jobs[job]) if job in jobs else None
    if not result:
        return jsonify(error='任务不存在'), 404
    if result['status'] == 'queued':
        result['ahead'] = _queue_position(job)
    if result['status'] == 'running':
        result['elapsed'] = round(time.time() - result['started'], 1)
    return jsonify(result)


@app.get('/api/jobs')
def list_jobs():
    with lock:
        items = [dict(id=j, **{k: s.get(k) for k in ('status', 'name', 'png', 'svg', 'preset', 'seconds', 'paths')})
                 for j, s in reversed(jobs.items())]
    return jsonify(items[:20])


@app.get('/file/<name>')
def file(name):
    if Path(name).name != name or Path(name).suffix not in ('.svg', '.png', '.source'):
        abort(404)
    base = UP if name.endswith('.source') else OUT
    path = base / name
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=request.args.get('download') == '1', download_name=name)


@app.get('/healthz')
def health():
    return jsonify(status='ok', weights=WEIGHTS.is_file(), presets=['auto', *PRESETS])
