"""AI line-art extraction (ControlNet annotator models by lllyasviel, Apache-2.0).
Network definitions copied from controlnet_aux so the project doesn't depend on that package.

  anime : lineart_anime  (netG.pth,     pix2pix U-Net, ~218 MB)
  sketch: lineart        (sk_model.pth, ResNet generator, ~17 MB)

line_map(rgb, model) -> float32 [0,1] "ink probability", 1 = line.
"""
import functools, math, numpy as np, cv2, torch, torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)



norm_layer = nn.InstanceNorm2d
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out




_cache = {}


def _load(kind, weights_dir="weights"):
    if kind in _cache:
        return _cache[kind]
    if kind == "anime":
        net = UnetGenerator(3, 1, 8, 64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False,
                                                                       track_running_stats=False), use_dropout=False)
        sd = torch.load(f"{weights_dir}/netG.pth", map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    else:
        net = Generator(3, 1, 3)
        sd = torch.load(f"{weights_dir}/sk_model.pth", map_location="cpu")
    net.load_state_dict(sd)
    net = net.eval().to(DEVICE)
    _cache[kind] = net
    return net


@torch.no_grad()
def line_map(rgb, kind="anime", detect_long=2048, weights_dir="weights"):
    """Run the line model at `detect_long` px on the long side (multiple of 256), return a
    [0,1] map at the input size where 1 = line."""
    net = _load(kind, weights_dir)
    H, W = rgb.shape[:2]
    s = detect_long / max(H, W)
    h, w = max(256, round(H * s / 256) * 256), max(256, round(W * s / 256) * 256)
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC).astype(np.float32)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEVICE)
    if kind == "anime":
        y = net(x / 127.5 - 1.0)[0, 0] * 0.5 + 0.5      # 1 = paper, 0 = ink
    else:
        y = net(x / 255.0)[0, 0]                       # 1 = paper, 0 = ink
    y = y.clamp(0, 1).cpu().numpy()
    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_LINEAR)
    return 1.0 - y


def ai_ink_mask(rgb, lines, rel_thr=0.06, win=21, line_thr=0.25, keep_frac=0.35, min_area=12, max_width=12):
    """Strokes = thin pixels darker than their local surroundings (any hue), kept only where the line
    model agrees. Component-level gating: a stroke component is kept if enough of it lies on
    model-detected lines, so thick outlines (which the model draws as two edges) survive whole,
    while shadow edges / texture the model ignores are dropped. Areas wider than max_width
    (pupils, dark fills) are not lines and stay in the colour layer."""
    from .color import srgb_to_oklab
    L = srgb_to_oklab(rgb)[..., 0].astype(np.float32)
    Lu8 = np.clip(L * 255, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(Lu8, win).astype(np.float32) / 255
    dark = ((med - L) > rel_thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width, max_width))
    blobs = cv2.dilate(cv2.morphologyEx(dark, cv2.MORPH_OPEN, k), np.ones((3, 3), np.uint8))
    dark = dark & (1 - blobs)
    on_line = cv2.dilate((lines > line_thr).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    hit = np.bincount(cc.ravel(), weights=on_line.ravel(), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = (hit / np.maximum(area, 1) >= keep_frac) & (area >= min_area); keep[0] = False
    return keep[cc]


def enforce_min_width(mask, min_px):
    """Printability: strokes whose average width is below min_px are thickened *as a whole*
    (uniform dilation per stroke), so lines stay even instead of getting lumps."""
    if min_px <= 1.5 or not mask.any():
        return mask
    from skimage.morphology import skeletonize
    m = mask.astype(np.uint8)
    n, cc = cv2.connectedComponents(m, connectivity=8)
    skel = skeletonize(m > 0)
    area = np.bincount(cc.ravel(), minlength=n).astype(np.float64)
    slen = np.bincount(cc[skel], minlength=n).astype(np.float64)
    width = area / np.maximum(slen, 1)                     # mean stroke width per component
    out = m > 0
    for r in range(1, int(math.ceil(min_px / 2)) + 1):     # group strokes by needed radius
        sel = np.zeros(n, bool); sel[1:] = np.ceil((min_px - width[1:]) / 2) == r
        if not sel.any():
            continue
        grp = sel[cc].astype(np.uint8)
        out |= cv2.dilate(grp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return out


def model_line_mask(rgb, lines, line_thr=0.4, fill_px=7, rel_thr=0.04, win=21, min_area=15):
    """Line SHAPE taken from the model output (smooth, continuous), not from thresholded pixels.
    The model draws thick strokes as two parallel edges; the space between them is filled where
    the original is dark (closing restricted to dark pixels), so wide outlines come out solid
    without merging neighbouring hair strands."""
    from .color import srgb_to_oklab
    edges = (cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 0.8) > line_thr).astype(np.uint8)
    L = srgb_to_oklab(rgb)[..., 0].astype(np.float32)
    med = cv2.medianBlur(np.clip(L * 255, 0, 255).astype(np.uint8), win).astype(np.float32) / 255
    dark = (med - L) > rel_thr
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_px, fill_px))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k) > 0
    m = (edges > 0) | (closed & dark)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[cc]


def stroke_mask(rgb, lines, rel_thr=0.06, win=21, line_thr=0.4, keep_frac=0.3, max_width=16, min_area=15):
    """Which pixels belong to drawn strokes: original pixels darker than their surroundings
    (true width), kept per component only where the model sees a line, plus the model's own
    line pixels to bridge gaps the original leaves (grain, anti-aliasing)."""
    from .color import srgb_to_oklab
    L = srgb_to_oklab(rgb)[..., 0].astype(np.float32)
    med = cv2.medianBlur(np.clip(L * 255, 0, 255).astype(np.uint8), win).astype(np.float32) / 255
    dark = ((med - L) > rel_thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width, max_width))
    dark &= 1 - cv2.dilate(cv2.morphologyEx(dark, cv2.MORPH_OPEN, k), np.ones((3, 3), np.uint8))   # not fills
    model = (cv2.GaussianBlur(lines.astype(np.float32), (0, 0), 0.8) > line_thr).astype(np.uint8)
    near = cv2.dilate(model, np.ones((5, 5), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    hit = np.bincount(cc.ravel(), weights=near.ravel(), minlength=n)
    keep = (hit / np.maximum(st[:, cv2.CC_STAT_AREA], 1) >= keep_frac); keep[0] = False
    m = keep[cc] | (model > 0)
    n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool); big[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return big[cc]


def redraw_strokes(mask, smooth_px=6.0, min_radius=0.0, ss=2):
    """Centreline + width redraw. Skeleton gives the path, the distance transform gives the local
    half-width, which is smoothed along the stroke; every skeleton point is redrawn as a disc of
    that radius (at `ss`x resolution). Result: even, smooth-edged strokes of the original width.
    Returns the supersampled mask (ss x size)."""
    from skimage.morphology import skeletonize
    from scipy import ndimage as ndi
    m = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    skel = skeletonize(m > 0)
    w = cv2.GaussianBlur((dt * skel).astype(np.float32), (0, 0), smooth_px)
    s = cv2.GaussianBlur(skel.astype(np.float32), (0, 0), smooth_px)
    r = np.where(skel, w / np.maximum(s, 1e-6), 0)
    r = np.maximum(r, min_radius)
    H, W = m.shape
    out = np.zeros((H * ss, W * ss), np.uint8)
    ys, xs = np.nonzero(skel)
    rr = r[ys, xs] * ss
    # bucket radii in 0.5 px steps (at ss) and stamp discs by dilation per bucket
    bins = np.round(rr * 2) / 2
    for b in np.unique(bins):
        sel = bins == b
        pts = np.zeros_like(out)
        pts[ys[sel] * ss + ss // 2, xs[sel] * ss + ss // 2] = 1
        rad = max(0.5, float(b))
        d = int(np.ceil(rad)) * 2 + 1
        disc = (np.hypot(*np.mgrid[-(d // 2):d // 2 + 1, -(d // 2):d // 2 + 1]) <= rad + 0.25).astype(np.uint8)
        out |= cv2.dilate(pts, disc)
    # light anti-staircase: blur + threshold at the supersampled scale
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), 0.7 * ss) > 0.5
    return out
