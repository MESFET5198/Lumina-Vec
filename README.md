# Lumina-Vec：面向叠色 3D 打印的 AI 辅助矢量化

把二次元插画转成 SVG，输出主要用于导入 Lumina-Layers 做多色叠层 3D 打印。目标是用开源组件加一个小型 AI 模型，达到接近 Vectorizer.AI 的效果，并且明暗过渡的边缘要平滑、能打印。

![对比](docs/full_qipao_shading.jpg)
![对比](docs/full_fox_shading.jpg)
![对比](docs/full_maid_shading.jpg)
<sub>每组从左到右：原图 · Vectorizer.AI · Lumina-Vec</sub>

## 流程

```
原图 ─► ① AI 去颗粒 ─► ② 均值漂移压平 ─► ③ 明暗分层 ─► ④ vtracer 描摹 ─► ⑤ 五官细节补回 ─► SVG
```

1. **AI 去颗粒**（`lumina_vec/denoise.py`）：先缩小 1/2，把颗粒平均掉；再用 Real-ESRGAN 动漫模型放大 4 倍，重新生成锐利的线稿。纯 CPU 运行。
2. **压平**：均值漂移滤波，把放大后残留的笔刷纹理压成平涂色块。
3. **明暗分层**（`lumina_vec/bands.py`）
   - **为什么需要**：vtracer 靠合并相似的相邻像素来长出区域。在平滑渐变上，这种合并会一路连下去，最后变成一整块纯色。所以必须先显式量化颜色。
   - **调色板**：在 OKLab 色彩空间做 k-means，取 112 色；另外为高对比像素单独分配 48 色，保证嘴、鼻子这些小线条有属于自己的颜色。
   - **平滑的色带边界**：在渐变区，像素的归属由一张"避开线稿的模糊图"决定。离线稿越远，模糊半径越大，最大到 σ=12；越靠近线稿半径越小，这样不会在下巴边缘留下一圈浅色晕边。最后再做一次空间投票。得到的色带边界是平滑的等值线，而不是锯齿状的阶梯，3D 打印时可以直接用。
   - **线稿和纹理保持原样**：强边缘上的像素保留逐像素的原始颜色。
   - **小碎块的处理**：低对比度的小碎块并入相邻区域；高对比度的小块（瞳孔高光、鼻点）保留。
3b. **轮廓线单独成层**（`lumina_vec/ink.py`，仅 `shading` 预设开启）：
   - **为什么需要**：重颗粒图经过 AI 去颗粒后，线稿边缘仍然毛糙，描出来的轮廓粗细不均、呈锯齿状。
   - **怎么做**：把细的、深色、低饱和的线条单独取出来。先 2 倍超采样，再高斯平滑、二值化，单独描摹后叠在色彩层之上。色彩层会延伸到线条底下，所以不会留缝。大面积的深色区域（比如黑发）不受影响，仍保留明暗。
   - **只用于重颗粒图**：干净的图线条本来就平滑，而且线条较细，平滑处理反而会把线吃掉。可以用 `--ink 0/1` 手动开关。
4. **描摹**：vtracer 1.0（stacked 分层 + 样条拟合）。
5. **五官细节补回**（`lumina_vec/overlay.py`）
   - **为什么需要**：vtracer 的层次聚类会把很小的色簇并入邻居。而一条 3 像素宽的嘴线，因为抗锯齿被拆成了约 10 种颜色、每种只有 1～3 像素，所以必然被吞掉。
   - **怎么做**：按"与局部背景的对比度"重新检测这类小块，每块统一成一种颜色，单独描摹后叠加在最上层。

## 结果

指标的计算方式：与"原图做 σ=2 高斯模糊后的版本"比较，ΔE 为 OKLab 色差 ×100。

| 图 | 工具 | ΔE↓ | SSIM↑ | 边缘 F1↑ | path 数 | 文件大小 |
|---|---|---|---|---|---|---|
| 旗袍：平涂，重度颗粒 | Vectorizer.AI | **1.93** | 0.898 | 0.809 | 1428 | 3.1 MB |
| | Lumina-Vec `shading` | 2.07 | 0.895 | **0.820** | 10463 | 2.9 MB |
| 狐耳：厚涂，纹理多 | Vectorizer.AI | **3.12** | **0.777** | 0.585 | 8588 | 15 MB |
| | Lumina-Vec `shading-detail` | 3.44 | 0.760 | **0.613** | 18322 | 4.1 MB |
| 女仆：干净的赛璐璐，3840×4800 | Vectorizer.AI | 2.50 | 0.892 | **0.604** | 3594 | 6.0 MB |
| | Lumina-Vec `shading-detail --work-scale 0.5` | **2.41** | **0.898** | 0.599 | 12295 | 2.7 MB |

**注意：指标会误导。** 最终参数都是靠局部肉眼比对选出来的。

| 旗袍腿部的明暗 | 旗袍面部的五官 | 狐耳的嘴 | 女仆面部的明暗 |
|---|---|---|---|
| ![](docs/shading_qipao_leg.jpg) | ![](docs/shading_qipao_face.jpg) | ![](docs/detail_fox_mouth.jpg) | ![](docs/shading_maid_face.jpg) |

### 主观评估
- **明暗过渡**：色带边界平滑，过渡自然，在旗袍和女仆这两张图上观感好于 Vectorizer.AI。
- **五官**：瞳孔高光、鼻点、嘴线都能保留下来。在狐耳这类五官极小的厚涂图上，鼻梁的高光线条仍会丢失。
- **代价**：path 数是 Vectorizer.AI 的 2～7 倍，主要来自色带和细节补回层；不过文件大小反而更小。
- **仍有差距**：小字（印章）是糊的；腮红变成了硬边色块；狐耳那张的眼睛内部细节偏少。

## 用法

```bash
pip install numpy pillow opencv-python-headless scikit-image scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --pre "vtracer==1.0.0a3"
./scripts/download_weights.sh

python -m lumina_vec.pipeline in.png out.svg                 # 默认 auto：自动选预设、自动缩放大图（推荐）
python -m lumina_vec.pipeline in.png out.svg --bands 48      # 减少颜色数（按打印需求选）
python -m lumina_vec.pipeline in.png out.svg --preset shading          # 手动：平涂二次元，颗粒重
python -m lumina_vec.pipeline in.png out.svg --preset shading-detail   # 手动：厚涂、细节多
python -m lumina_vec.pipeline in.png out.svg --preset illustration     # 不分层：色块最少、文件最小
```

**在两台机器上分工跑（AI 步骤放到 N 卡主机上）**

```bash
# GPU 主机：需要安装 CUDA 版 torch（pip install torch --index-url https://download.pytorch.org/whl/cu121）
python -m lumina_vec.denoise in.png in_clean.png          # 有 CUDA 时自动使用 GPU + fp16
# 主力机：跳过 AI 步骤，直接用 GPU 主机上处理好的结果
python -m lumina_vec.pipeline in.png out.svg --clean in_clean.png
```

`--clean` 只替换 AI 去颗粒这一步。auto 模式仍然根据原图判断颗粒程度，所以第一个参数必须传原图。

**各步骤耗时**（狐耳图，1600×2460，2 核云容器）：Real-ESRGAN 约 14 秒、均值漂移约 5 秒、明暗分层约 20 秒（之前约 70 秒，已优化）、vtracer 约 20 秒、细节补回约 2 秒，总计约 67 秒（之前约 124 秒）。

**auto 模式的判断规则**
- **颗粒程度**：原图灰度与其 3×3 中值滤波结果之间的平均差。≥ 4 时用 `shading`，否则用 `shading-detail`。三张测试图的值分别是：旗袍 8.8、狐耳 1.9、女仆 0.2。
- **工作分辨率**：`shading` 的上限是 1400 万像素，`shading-detail` 的上限是 600 万像素，超过上限的图会自动缩小。
- 选择结果会打印出来；单独指定的参数（例如 `--bands`、`--work-scale`）会覆盖 auto 的选择。
- 阈值只用三张图标定过。遇到风格明显不同的图，自动选择未必合适，可以手动指定 `--preset`。

主要参数：

| 参数 | 作用 |
|---|---|
| `bands` | 明暗分层的颜色数，0 表示关闭 |
| `blur_sigma` | 渐变区的最大模糊半径：越大，色带边界越平滑 |
| `vote_sigma` | 空间投票的半径 |
| `l_weight` / `c_weight` | k-means 中明度和色度的权重。明度权重过高时，浅色皮肤会和白色归为一类，导致发灰 |
| `edge_thr` | 高于这个边缘强度的像素保留原始颜色（保护线稿和纹理） |
| `filter_speckle` | vtracer 的碎块过滤阈值，按边长计。设成 12 会吃掉瞳孔高光和鼻子 |
| `pre_down` / `work_scale` | AI 处理前的缩小倍数 / 描摹时的工作分辨率倍数 |

评估工具（需要 Node + Playwright）：

```bash
python -m lumina_vec.evaluate original.png a.svg b.svg
LVEC_REGIONS='{"face":[x0,y0,x1,y1]}' python -m lumina_vec.crops docs/cmp original.png a.svg.png b.svg.png
```

## 实验记录

| 方案 | 结果 |
|---|---|
| OKLab k-means + 众数滤波 + 无缝镶嵌 | 指标接近 Vectorizer.AI，但线稿碎裂 |
| 调低 vtracer 的 `layer_difference` 来保留渐变 | 无效：渐变会被链式合并成一块 |
| k-means 分层 + 全局平滑 | 有明暗过渡了，但毛衣纹理和发丝被抹掉 |
| 只在低对比度区平滑 | 纹理保住了，但色带边缘呈锯齿状，没法打印 |
| 用均值漂移后的图来定色带边界 | 均值漂移产生的平台是块状的，色带边界也跟着成块 |
| 固定 σ 的模糊来定色带边界 | 边界平滑了，但线稿会渗进皮肤，下巴边缘出现一圈浅色晕边 |
| **避开线稿、按距离调节模糊半径**（当前方案） | 色带平滑，也没有晕边 |
| `filter_speckle=12` | 瞳孔高光、鼻子、嘴都被删掉 → 改为 4 |
| 只按颜色区域的大小来找五官 | 抗锯齿把嘴拆成约 10 种颜色，每块只有 1～3 像素，检测不到 → 改为按对比度检测 |

## 依赖与许可
- [vtracer](https://github.com/visioncortex/vtracer)：MIT
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 的 `realesr-animevideov3` 权重：BSD-3-Clause（不随仓库分发，需运行 `scripts/download_weights.sh` 下载）
- 测试原图和 Vectorizer.AI 的输出不包含在仓库里，`docs/` 中只有比对截图
