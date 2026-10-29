# aivec：AI 辅助的位图转 SVG 矢量化

目标：用开源组件加一个小型 AI 模型，在**带噪点的二次元插画**上做到 Vectorizer.AI 约 80% 的效果。

![对比](docs/full_compare.jpg)
<sub>左：原图（带胶片颗粒）· 中：Vectorizer.AI · 右：aivec（`illustration` 预设）</sub>

## 流程

```
原图 ──► ① AI 去颗粒 ──► ② 均值漂移压平 ──► ③ vtracer 分层聚类 + 样条拟合 ──► SVG
```

1. **AI 去颗粒**（`aivec/denoise.py`）：先按 1/2 盒式缩小（把颗粒平均掉），再用 Real-ESRGAN 动漫模型 `realesr-animevideov3` 放大 4 倍，重新生成锐利的线稿，最后重采样到 1.5 倍工作分辨率。纯 CPU 运行，1741×3379 的图约 20 秒。
2. **压平**：用 `cv2.pyrMeanShiftFiltering` 把放大模型留下的"笔刷纹理"压成平涂色块，否则镜片、头发上会出现斑块。
3. **描摹**：vtracer 1.0（stacked 分层聚类 + 样条拟合 + `simplify`）。
4. 输出 SVG 时保持原图像素尺寸，`viewBox` 保留工作分辨率。

## 结果（测试图：1741×3379 二次元插画，重度颗粒）

| | ΔE↓ | SSIM↑ | 边缘 F1↑ | path 数 | 文件大小 |
|---|---|---|---|---|---|
| Vectorizer.AI | **1.93** | **0.898** | 0.809 | 1428 | 3.1 MB |
| vtracer 直接处理原图 | 2.74 | 0.710 | 0.286 | 73,904 | 26 MB |
| **aivec** | 3.19 | 0.895 | **0.845** | **1063** | **0.69 MB** |

指标的计算方式：与"原图做 σ=2 高斯模糊后的版本"比较，ΔE 为 OKLab 色差 ×100。结构和边缘已与 Vectorizer.AI 持平，色差仍明显落后（见下方"差距"）。

**注意：指标会误导。** 实验中有几个版本指标几乎追平 Vectorizer.AI，肉眼看却明显更差（线稿断裂、色块斑驳）。所以最终方案是靠局部肉眼比对选出来的，比对图在 `docs/`。

| 面部 | 手 | 腿部阴影 |
|---|---|---|
| ![](docs/cmp_face.jpg) | ![](docs/cmp_hand.jpg) | ![](docs/cmp_leg.jpg) |

每张图从左到右依次是：原图 · Vectorizer.AI · vtracer 直接处理 · aivec。

### 主观评估
- **接近或持平**：线稿干净度、手部、服装大色块、镜片（比 Vectorizer.AI 更干净，但丢了镜片上的反光色带）、虹膜颜色（保留了紫色，Vectorizer.AI 偏棕）
- **明显差距**：
  - 皮肤的柔和阴影被压平。Vectorizer.AI 会保留几层渐变色带，aivec 基本只剩一个颜色，这也是 ΔE 落后的主要原因。
  - 小字（印章上的"珍""羞"）糊成一团。问题出在去颗粒这一步：先缩小 2 倍会丢掉约 4px 宽的笔画，而 1 倍缩小又会残留颗粒。
  - 背景的大面积渐变被压成纯色。
- 整体估计：在这张图上大约达到 Vectorizer.AI 的 **75–80%**。目前**只在一张图上调过参**，其他画风可能需要换预设。

## 用法

```bash
pip install numpy pillow opencv-python-headless scikit-image
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --pre "vtracer==1.0.0a3"
./scripts/download_weights.sh

python -m aivec.pipeline input.png output.svg                  # 默认 illustration 预设
python -m aivec.pipeline input.png output.svg --preset flat    # 色块更少、更大，会丢掉细小笔画
python -m aivec.pipeline input.png output.svg --preset detail  # 保留更多细节，文件更大
python -m aivec.pipeline input.png output.svg --filter-speckle 8 --simplify 1.0   # 单独覆盖参数

# 评估（需要 Node + Playwright Chromium）
python -m aivec.evaluate original.png a.svg b.svg
python -m aivec.crops docs/cmp original.png a.svg.png b.svg.png
```

主要参数：

| 参数 | 作用 |
|---|---|
| `pre_down` | AI 处理前的缩小倍数：越大去噪越强，但小细节丢得越多 |
| `work_scale` | 描摹时的分辨率倍数（相对原图） |
| `ms_sp` / `ms_sr` | 均值漂移的空间半径和颜色半径：越大越平 |
| `filter_speckle` | 丢弃小于该面积的碎块：太大会吃掉嘴、腮红这类细笔画 |
| `layer_difference` | 分层的颜色差阈值：越小保留的渐变层越多 |
| `simplify` | 曲线简化容差（px） |

## 试过但放弃的方案（实验记录）

| 方案 | 结果 |
|---|---|
| OKLab k-means（88 色）+ 众数滤波 + 无缝镶嵌 | 指标接近 Vectorizer.AI，但线稿碎裂、阴影边缘呈锯齿状 |
| 线稿单独分层，色彩层在线下填补后强力平滑 | 线稿问题解决了，但嘴、腮红、虹膜这类有颜色的细笔画被抹掉 |
| black-hat 检测细笔画并单独分层 | 能找回细笔画，但用颜色键背景描摹时，细线会和键色混合变灰 |
| 每种颜色单独做二值描摹 | 相邻颜色之间出现缝隙，边缘参差 |
| 按渲染像素给每条 path 重新取色（color refit） | 几乎没有改善：误差来自区域形状被合并，不是填充色不对 |
| 频率压平采样的 k-means | 稀有颜色（紫色虹膜）能被保留下来，可以留作后续的改进方向 |

## 下一步可以做的
- **皮肤渐变**：只对低饱和、大面积的区域做分层量化，再叠加到 vtracer 的结果上
- **小字**：去颗粒改为按局部内容自适应选择 `pre_down`，或者换用 1x 去噪模型（例如 SCUNet）
- 在更多不同画风的图上验证、调整预设

## 依赖与许可
- [vtracer](https://github.com/visioncortex/vtracer)：MIT
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 的 `realesr-animevideov3` 权重：BSD-3-Clause（不随仓库分发，需运行 `scripts/download_weights.sh` 下载）
- 测试图和 Vectorizer.AI 的输出不包含在仓库里，`docs/` 中只有比对截图
