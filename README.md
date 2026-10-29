# aivec：AI 辅助的位图转 SVG 矢量化

目标：用开源组件加一个小型 AI 模型，在二次元插画上做到接近 Vectorizer.AI 的效果。输出主要用于导入 Lumina-Layers，做多色叠层 3D 打印。

![对比](docs/full_qipao_shading.jpg)
<sub>左：原图（重度胶片颗粒）· 中：Vectorizer.AI · 右：aivec `shading` 预设</sub>

![对比](docs/full_fox_shading.jpg)
<sub>左：原图（厚涂、细节多）· 中：Vectorizer.AI · 右：aivec `shading-detail` 预设</sub>

## 流程

```
原图 ─► ① AI 去颗粒 ─► ② 均值漂移压平 ─► ②b 明暗分层（可选）─► ③ vtracer 描摹 ─► SVG
```

1. **AI 去颗粒**（`aivec/denoise.py`）：先按 1/2 盒式缩小，把颗粒平均掉；再用 Real-ESRGAN 动漫模型 `realesr-animevideov3` 放大 4 倍，重新生成锐利的线稿；最后重采样到工作分辨率。纯 CPU 即可运行。
2. **压平**：用 `cv2.pyrMeanShiftFiltering` 把放大模型留下的"笔刷纹理"压成平涂色块。
3. **明暗分层**（`aivec/bands.py`，`shading` 系列预设）：
   - **为什么需要**：vtracer 靠"合并相似的相邻像素"来长出区域。平滑渐变上每个像素都和旁边的很像，会一路链式合并成一整块纯色，阈值调多低都没用。所以必须**先显式量化**颜色。
   - **怎么做**：
     - 在 OKLab 色彩空间做 k-means，默认 112 色。采样时压平颜色频率，让虹膜、腮红这类面积小但颜色独特的区域也能分到色位；明度权重 1.3，色度权重 2.0。
     - 空间平滑投票**只作用于低对比度像素**（渐变区）。强边缘上（线稿、毛衣针织纹理）保留逐像素的原始标签，细节不会被抹掉。
     - 每个色块取原图中对应像素的中位色。
   - **结果**：色带边界沿平滑后图像的等值线走，形态接近 Vectorizer.AI 的明暗分层。
4. **描摹**：vtracer 1.0（stacked 分层 + 样条拟合 + `simplify`）。输出 SVG 时保持原图像素尺寸。

## 结果

指标的计算方式：与"原图做 σ=2 高斯模糊后的版本"比较，ΔE 为 OKLab 色差 ×100。

**图 1**：1741×3379 平涂二次元，重度颗粒

| | ΔE↓ | SSIM↑ | 边缘 F1↑ | path 数 | 颜色数 | 文件大小 |
|---|---|---|---|---|---|---|
| Vectorizer.AI | **1.93** | 0.898 | 0.809 | 1428 | 88 | 3.1 MB |
| vtracer 直接处理原图 | 2.74 | 0.710 | 0.286 | 73,904 | – | 26 MB |
| aivec `illustration` | 3.19 | 0.895 | **0.845** | 1063 | – | 0.69 MB |
| **aivec `shading`** | 2.08 | **0.900** | 0.791 | 2676 | 112 | 1.4 MB |

**图 2**：1600×2460 厚涂插画，轻颗粒，纹理多

| | ΔE↓ | SSIM↑ | 边缘 F1↑ | path 数 | 颜色数 | 文件大小 |
|---|---|---|---|---|---|---|
| Vectorizer.AI | **3.12** | 0.777 | 0.585 | 8588 | 145 | 15 MB |
| aivec `detail` | 4.19 | 0.764 | **0.608** | 5208 | – | 1.7 MB |
| **aivec `shading-detail`** | 3.36 | **0.778** | 0.577 | 5197 | 111 | 2.2 MB |

**注意：指标会误导。** 实验中有几个版本指标几乎追平 Vectorizer.AI，肉眼看却明显更差（线稿断裂、色块斑驳、肤色发灰）。所以最终参数是靠局部肉眼比对选出来的，比对图在 `docs/`。

| 图 1 腿部明暗 | 图 2 毛衣明暗 | 图 2 面部 |
|---|---|---|
| ![](docs/shading_qipao_leg.jpg) | ![](docs/shading_fox_sweater.jpg) | ![](docs/shading_fox_face.jpg) |

### 主观评估
- **`shading` 对比 `illustration`**：皮肤、毛衣、背景的明暗过渡从"一整块纯色"变成了多层色带，接近 Vectorizer.AI；代价是 path 数和颜色数增加，色带边缘偶尔有阶梯状锯齿。
- **持平或更好**：线稿、手部、虹膜颜色（图 1 保留了紫色）、毛衣的针织纹理（图 2 中比 Vectorizer.AI 更完整）。
- **仍有差距**：
  - 小字（图 1 的印章）是糊的；
  - 图 2 眼睛内部的细碎高光较少；
  - 腮红变成了一块硬边色块，没有过渡；
  - 低饱和背景区域偶尔出现杂色色带。
- **3D 打印（Lumina）**：色带越多，叠层越细，但区域也越碎。可以用 `--bands` 控制颜色数，例如 `--bands 48`。

## 用法

```bash
pip install numpy pillow opencv-python-headless scikit-image
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --pre "vtracer==1.0.0a3"
./scripts/download_weights.sh

python -m aivec.pipeline in.png out.svg --preset shading          # 平涂/二次元，带明暗分层（推荐）
python -m aivec.pipeline in.png out.svg --preset shading-detail   # 厚涂/细节多的插画
python -m aivec.pipeline in.png out.svg --preset illustration     # 不分层：色块最少、文件最小
python -m aivec.pipeline in.png out.svg --preset shading --bands 48   # 减少颜色数
```

| 预设 | 适合 | 说明 |
|---|---|---|
| `shading` | 平涂二次元，颗粒重 | 1.5 倍工作分辨率 + 明暗分层 |
| `shading-detail` | 厚涂、纹理多、颗粒轻 | 1 倍工作分辨率 + 明暗分层 |
| `illustration` | 想要最少的色块 | 不分层，渐变会被压平 |
| `flat` | 更少、更大的色块 | 会丢掉嘴、腮红这类细笔画 |
| `detail` | 不分层但多保留细节 | |

主要参数：

| 参数 | 作用 |
|---|---|
| `bands` | 明暗分层的颜色数，0 表示关闭。越多过渡越细，区域也越碎 |
| `l_weight` / `c_weight` | k-means 中明度和色度的权重。明度权重过高时，浅色皮肤会和白色归为一类，导致发灰 |
| `edge_thr` | 高于这个边缘强度的像素保留原始标签（保护线稿和纹理） |
| `vote_sigma` | 渐变区空间投票的半径：越大色带边缘越圆滑 |
| `pre_down` | AI 处理前的缩小倍数：越大去噪越强，但小细节丢得越多 |
| `work_scale` | 描摹时的分辨率倍数 |
| `ms_sp` / `ms_sr` | 均值漂移的空间半径和颜色半径 |
| `filter_speckle` | 丢弃小于该面积的碎块 |
| `simplify` | 曲线简化容差（px） |

评估工具（需要 Node + Playwright Chromium）：

```bash
python -m aivec.evaluate original.png a.svg b.svg
AIVEC_REGIONS='{"face":[x0,y0,x1,y1]}' python -m aivec.crops docs/cmp original.png a.svg.png b.svg.png
```

## 实验记录

| 方案 | 结果 |
|---|---|
| OKLab k-means + 众数滤波 + 无缝镶嵌 | 指标接近 Vectorizer.AI，但线稿碎裂、阴影边缘锯齿 |
| 线稿单独分层，色彩层强力平滑 | 嘴、腮红、虹膜这类有颜色的细笔画被抹掉 |
| black-hat 检测细笔画并单独分层，每种颜色单独二值描摹 | 细线和键色混合变灰；相邻颜色之间出现缝隙 |
| 按渲染像素给每条 path 重新取色 | 几乎没有改善：误差来自区域形状被合并，不是填充色不对 |
| 调低 vtracer 的 `layer_difference` 来保留渐变 | 无效：渐变会被链式合并成一块 |
| k-means 分层 + 全局平滑 | 有明暗过渡了，但毛衣纹理和发丝被抹掉 |
| **分层 + 只在低对比度区平滑**（当前方案） | 明暗过渡和细节兼顾 |
| 明度权重 1.6～2.0 | 浅色皮肤和白色背景/毛衣归为一类，肤色发灰 → 改为明度 1.3、色度 2.0 |

## 依赖与许可
- [vtracer](https://github.com/visioncortex/vtracer)：MIT
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 的 `realesr-animevideov3` 权重：BSD-3-Clause（不随仓库分发，需运行 `scripts/download_weights.sh` 下载）
- 测试图和 Vectorizer.AI 的输出不包含在仓库里，`docs/` 中只有比对截图
