import io, json, os, re, threading, time, uuid
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_file, abort
from lumina_vec.pipeline import PRESETS, auto_params, fine_params, grain_level, vectorize
import cairosvg

ROOT = Path(os.environ.get('LVEC_DATA', '/data')); UP = ROOT / 'uploads'; OUT = ROOT / 'outputs'
WEIGHTS = Path(os.environ.get('LVEC_WEIGHTS', '/app/weights/realesr-animevideov3.pth'))
INDEX = Path(__file__).with_name('index.html')
UP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__); app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

jobs = OrderedDict()            # job id -> state dict (newest last); persisted to JOBS_FILE
lock = threading.Lock()
runner = threading.Semaphore(1)  # CPU-heavy: run one conversion at a time, queue the rest
MAX_PIXELS = 40_000_000
JOBS_FILE = ROOT / 'jobs.json'
VOLATILE = ('progress', 'stage')   # live-only fields, not worth a disk write per update


def _save():
    """Write the job list to /data (call with `lock` held). Atomic: tmp file + rename."""
    tmp = JOBS_FILE.with_suffix('.tmp')
    data = {j: {k: v for k, v in s.items() if k not in VOLATILE} for j, s in jobs.items()}
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding='utf8')
    os.replace(tmp, JOBS_FILE)


def _set(job, **kw):
    with lock:
        jobs[job].update(kw)
        if set(kw) - set(VOLATILE):
            _save()


def _load():
    """Job history survives restarts: read /data/jobs.json, mark jobs a restart cut off as
    interrupted, and rebuild records for result files that have none (older versions kept the
    list in memory only)."""
    data = {}
    if JOBS_FILE.is_file():
        try:
            data = json.loads(JOBS_FILE.read_text(encoding='utf8'))
        except (OSError, ValueError):
            JOBS_FILE.rename(JOBS_FILE.with_suffix('.broken'))
    for j, s in data.items():
        if s.get('status') in ('queued', 'running'):
            s.update(status='interrupted', error='服务重启，任务中断')
    for svg in OUT.glob('*.svg'):
        j = svg.stem
        if j in data:
            continue
        rec = dict(status='done', name=f'{j[:8]}（恢复）', recovered=True, created=svg.stat().st_mtime,
                   finished=svg.stat().st_mtime, svg=f'/file/{j}.svg', bytes=svg.stat().st_size)
        png = OUT / f'{j}.png'
        if png.is_file():
            rec['png'] = f'/file/{j}.png'
            try:
                with Image.open(png) as im:
                    rec['width'], rec['height'] = im.size
            except OSError:
                pass
        try:
            txt = svg.read_text(encoding='utf8', errors='ignore')
            rec['paths'] = txt.count('<path')
        except OSError:
            pass
        data[j] = rec
    for src in UP.glob('*.source'):
        if src.stem not in data:
            data[src.stem] = dict(status='interrupted', name=f'{src.stem[:8]}（恢复）', recovered=True,
                                  created=src.stat().st_mtime, error='没有结果文件')
    for j, s in sorted(data.items(), key=lambda kv: kv[1].get('created', 0)):
        jobs[j] = s
    with lock:
        _save()


def _queue_position(job):
    with lock:
        waiting = [j for j, s in jobs.items() if s['status'] == 'queued']
    return waiting.index(job) if job in waiting else 0


def work(job, raw, preset, overrides, fine=False):
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
            if fine:
                params = fine_params(im, params)
            params.update(overrides)
            params['fine'] = fine
            t = time.monotonic()
            def prog(frac, label):           # pipeline = 0-92 %, PNG preview render = last 8 %
                _set(job, progress=round(frac * 0.92, 4), stage=label)
            svg, _ = vectorize(im, str(WEIGHTS), ai=True, progress=prog,
                               **{k: v for k, v in params.items() if k != 'fine'})
            _set(job, progress=0.93, stage='生成 PNG 预览')
            secs = time.monotonic() - t
            svgpath = OUT / f'{job}.svg'; pngpath = OUT / f'{job}.png'
            svgpath.write_text(svg, encoding='utf8')
            cairosvg.svg2png(bytestring=svg.encode(), write_to=str(pngpath), output_width=w, output_height=h)
            colors = len(set(m.lower() for m in re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg)))
            _set(job, status='done', progress=1.0, stage='完成', svg=f'/file/{job}.svg', png=f'/file/{job}.png', width=w, height=h,
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
    pmm = request.form.get('print_mm', '').strip()
    if pmm:
        try:
            pmm = float(pmm); lmm = float(request.form.get('line_mm', '0.42') or 0.42)
        except ValueError:
            return jsonify(error='打印尺寸或线宽格式不对'), 400
        if not (20 <= pmm <= 2000) or not (0.1 <= lmm <= 2):
            return jsonify(error='打印尺寸需在 20–2000 mm，线宽 0.1–2 mm'), 400
        overrides.update(print_mm=pmm, line_mm=lmm)
        if 'bands' not in overrides:
            overrides['bands'] = 56
    if request.form.get('lines', '').strip() == 'none':
        overrides['lines'] = None
    ink = request.form.get('ink', '').strip()
    if ink in ('0', '1'):
        overrides['ink'] = ink == '1'
    fine = request.form.get('fine', '').strip() == '1'
    ls = request.form.get('line_style', '').strip()
    if ls in ('v3', 'contour'):
        overrides['line_style'] = ls
    thin = request.form.get('thin', '').strip()
    if thin in ('0', '1'):
        overrides['thin_thr'] = 0.015 if thin == '1' else 0.0
    tracer = request.form.get('tracer', '').strip()
    if tracer in ('contour', 'vtracer'):
        overrides['tracer'] = tracer
    raw = up.read(); job = uuid.uuid4().hex
    (UP / f'{job}.source').write_bytes(raw)
    with lock:
        jobs[job] = {'status': 'queued', 'name': Path(up.filename).name[:120], 'created': time.time(),
                     'request': dict(preset=preset, overrides=overrides, fine=fine)}
        _save()
    threading.Thread(target=work, args=(job, raw, preset, overrides, fine), daemon=True).start()
    return jsonify(job=job, status='queued')


@app.post('/api/jobs/<job>/retry')
def retry(job):
    """Run an interrupted or failed job again with the settings it was submitted with."""
    src = UP / f'{job}.source'
    with lock:
        s = jobs.get(job)
        if not s:
            return jsonify(error='任务不存在'), 404
        if s['status'] in ('queued', 'running'):
            return jsonify(error='任务还在进行'), 409
        req = s.get('request')
        if not req or not src.is_file():
            return jsonify(error='缺少原图或转换设置，无法重试'), 409
        for k in ('error', 'svg', 'png', 'started', 'finished', 'seconds', 'paths', 'params'):
            s.pop(k, None)
        s.update(status='queued', created=time.time())
        jobs.move_to_end(job)
        _save()
    threading.Thread(target=work, args=(job, src.read_bytes(), req['preset'], req['overrides'], req['fine']),
                     daemon=True).start()
    return jsonify(job=job, status='queued')


@app.delete('/api/jobs/<job>')
def delete_job(job):
    """Delete a job's record and its files (original upload, SVG, PNG preview)."""
    with lock:
        s = jobs.get(job)
        if not s:
            return jsonify(error='任务不存在'), 404
        if s['status'] in ('queued', 'running'):
            return jsonify(error='任务还在进行，不能删除'), 409
        jobs.pop(job)
        _save()
    for f in (UP / f'{job}.source', OUT / f'{job}.svg', OUT / f'{job}.png'):
        f.unlink(missing_ok=True)
    return jsonify(ok=True)


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
    limit = min(max(request.args.get('limit', 200, type=int), 1), 1000)
    with lock:
        items = [dict(id=j, retry=bool(s.get('request')),
                      **{k: s.get(k) for k in ('status', 'name', 'png', 'svg', 'preset', 'seconds', 'paths',
                                               'error', 'created')})
                 for j, s in reversed(jobs.items())]
    return jsonify(items[:limit])


@app.get('/file/<name>')
def file(name):
    if Path(name).name != name or Path(name).suffix not in ('.svg', '.png', '.source'):
        abort(404)
    base = UP if name.endswith('.source') else OUT
    path = base / name
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=request.args.get('download') == '1', download_name=name)


_load()


@app.get('/healthz')
def health():
    return jsonify(status='ok', weights=WEIGHTS.is_file(), presets=['auto', *PRESETS])
