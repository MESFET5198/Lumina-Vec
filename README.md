# Lumina-Vec

把二次元插画转成 SVG，用于导入 [Lumina-Layers](https://github.com/lumina-layer-studio/Lumina-Layers) 做多色叠层 3D 打印。处理流程是：AI 去颗粒 → 明暗分层 → AI 线条层 → vtracer 描摹，另有专门针对打印的清理模式。

![示例](docs/full_qipao_shading.jpg)

## 快速开始

**Docker（含 Web UI）**

```bash
docker compose up -d --build      # 构建时会自动下载两个模型（约 220 MB）
# 浏览器打开 http://<主机>:8091
```

Web UI 支持拖拽或粘贴上传、预设选择、打印尺寸，可以拖动对比线、滚轮放大检查结果，会显示转换进度，并保留最近的转换记录。

**命令行**

```bash
pip install numpy pillow opencv-python-headless scikit-image scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu      # 有 N 卡就换成 CUDA 版
pip install --pre "vtracer==1.0.0a3"
./scripts/download_weights.sh

python -m lumina_vec.pipeline in.png out.svg                           # 自动选预设
python -m lumina_vec.pipeline in.png out.svg --print-mm 263            # 打印模式
```

## 预设

默认是 `auto`，会根据图片自动选择：

| 预设 | 自动选择的条件 | 适合的图 | 线条层 |
|---|---|---|---|
| `shading` | 颗粒度 ≥ 4 | 带颗粒、噪点的平涂图 | 开 |
| `cel` | 平缓渐变占比 < 0.18 | 干净的平涂赛璐璐 | 开 |
| `shading-detail` | 其余情况 | 厚涂、带渐变的插画 | 关 |

另外还有 `illustration`、`detail`、`flat` 三个不分层的旧预设，只能手动指定。大图会自动缩小处理，`shading` 上限 1400 万像素，其余 600 万像素。判断阈值只用 6 张测试图标定过，效果不对时可以手动指定 `--preset`。

## 3D 打印模式

```bash
# 只打印一部分时：先裁切，--print-mm 填裁切后的图在打印时长边的尺寸
python -m lumina_vec.pipeline crop.png out.svg --print-mm 263 --line-mm 0.42
```

- 比一根挤出线还窄的色块和碎块会并入相邻颜色，画面保持完整、不留空洞；
- 细线（轮廓、发丝、嘴）不删，而是整条加粗到线宽，所以线条会比原图略粗；
- 颜色数降到 56，描摹改用共边模式；
- 预览用的遮缝描边导入 Lumina 时会被自动删掉。

实测同一张图：预估打印时间从 10h04m 降到 8h56m，周长小于 2 mm 的外墙小圈减少约一半。切片端把回抽最小距离调到 4.5 mm 后，降到 8h28m。

检查工具：

```bash
python -m lumina_vec.printcheck out.svg --print-mm 263       # 切片前：统计比挤出点还小的区域
python -m lumina_vec.gcodecheck a.gcode b.gcode               # 切片后：对比小圈数、空驶、回抽和预估时间
```

## 常用参数

| 参数 | 作用 |
|---|---|
| `--preset` | `auto` / `shading` / `shading-detail` / `cel` / … |
| `--bands N` | 颜色数，越少打印出来的区域越少 |
| `--print-mm` / `--line-mm` | 打印尺寸（长边，mm）/ 挤出线宽，默认 0.42 |
| `--ink 0/1` | 强制关闭或开启线条层 |
| `--lines none` | 线条层不用 AI 模型，改按颜色规则找线 |
| `--work-scale` | 处理时的分辨率倍数 |
| `--clean img` | 使用其他机器上已做完 AI 去颗粒的图（见下） |

**把 AI 步骤放到 GPU 主机上跑**：先在 GPU 主机上运行 `python -m lumina_vec.denoise in.png clean.png`，再在主力机上运行 `python -m lumina_vec.pipeline in.png out.svg --clean clean.png`。不过 AI 步骤只占总耗时的约 20%，整体放到 GPU 主机上跑效果更明显。

## 已知限制

- 印章小字这类小于 2 mm 的细节会糊掉；腮红会变成硬边色块；
- 带色差特效（红青描边）的图，打印模式下青色边可能会丢失；
- 打印模式下，线条与色块交界处会残留零星 1～2 像素的点。按像素统计会算进碎块数，但实际打印影响很小；
- 实现细节、实测数据和试过但放弃的方案，见 [docs/NOTES.md](docs/NOTES.md)。

## 依赖与许可

- [vtracer](https://github.com/visioncortex/vtracer)：MIT
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesr-animevideov3`：BSD-3-Clause
- [ControlNet Annotators](https://huggingface.co/lllyasviel/Annotators) `lineart_anime`（`netG.pth`）：许可见上游仓库
- 模型权重不随仓库分发，需要单独下载；测试原图和参考 SVG 不包含在仓库中
