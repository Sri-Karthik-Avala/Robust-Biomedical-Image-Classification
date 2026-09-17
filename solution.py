import os, sys, time, math, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RES = 160


def resolve_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = ['.', 'public', './public', 'dataset', './dataset', '..', '../public',
             here, os.path.join(here, 'public'), '/kaggle/input']
    extra = []
    for r in list(roots):
        try:
            extra += [p for p in glob.glob(os.path.join(r, '*')) if os.path.isdir(p)]
        except Exception:
            pass
    roots = roots + extra

    def hunt(names):
        for r in roots:
            for n in names:
                p = os.path.join(r, n)
                if os.path.exists(p):
                    return p
        for r in roots:
            for n in names:
                hits = glob.glob(os.path.join(r, '**', n), recursive=True)
                if hits:
                    return hits[0]
        return None

    return {
        'train_csv': hunt(['train.csv']),
        'classes_csv': hunt(['classes.csv']),
        'train_npz': hunt([os.path.join('train', 'images.npz'), 'train_images.npz']),
        'test_npz': hunt([os.path.join('test', 'images.npz'), 'test_images.npz']),
        'test_meta': hunt([os.path.join('test', 'metadata.csv'), 'metadata.csv']),
    }


def load_npz_images(path):
    d = np.load(path)
    key = 'images' if 'images' in d.files else d.files[0]
    return d[key]


def hue_matrix(theta):
    c = torch.cos(theta); s = torch.sin(theta); B = theta.shape[0]
    M = torch.empty(B, 3, 3, device=theta.device)
    M[:, 0, 0] = 0.213 + c * 0.787 - s * 0.213
    M[:, 0, 1] = 0.715 - c * 0.715 - s * 0.715
    M[:, 0, 2] = 0.072 - c * 0.072 + s * 0.928
    M[:, 1, 0] = 0.213 - c * 0.213 + s * 0.143
    M[:, 1, 1] = 0.715 + c * 0.285 + s * 0.140
    M[:, 1, 2] = 0.072 - c * 0.072 - s * 0.283
    M[:, 2, 0] = 0.213 - c * 0.213 - s * 0.787
    M[:, 2, 1] = 0.715 - c * 0.715 + s * 0.715
    M[:, 2, 2] = 0.072 + c * 0.928 + s * 0.072
    return M


def gauss_kernel(sigma, ks, dev):
    ax = torch.arange(ks, device=dev) - ks // 2
    k = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
    return k / k.sum()


def colormap_remap(x):
    B, _, H, W = x.shape; d = x.device
    lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0, 1)
    K = 5
    cps = torch.rand(B, 3, K, device=d)
    t = lum * (K - 1)
    lo = t.floor().clamp(0, K - 2).long()
    fr = (t - lo.float())
    out = []
    for ch in range(3):
        c0 = torch.gather(cps[:, ch], 1, lo.view(B, -1)).view(B, 1, H, W)
        c1 = torch.gather(cps[:, ch], 1, (lo + 1).view(B, -1)).view(B, 1, H, W)
        out.append(c0 * (1 - fr) + c1 * fr)
    return torch.cat(out, 1)


def upscale(x):
    if x.shape[-1] != RES:
        x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    return x


def augment(x):
    B, _, H, W = x.shape; d = x.device
    ang = (torch.rand(B, device=d) - 0.5) * 2 * 0.30
    sc = torch.exp((torch.rand(B, device=d) - 0.5) * 2 * 0.30)
    tx = (torch.rand(B, device=d) - 0.5) * 0.30
    ty = (torch.rand(B, device=d) - 0.5) * 0.30
    fx = torch.where(torch.rand(B, device=d) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(B, device=d) < 0.25, -1.0, 1.0)
    ca, sa = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=d)
    theta[:, 0, 0] = ca / sc * fx; theta[:, 0, 1] = -sa / sc; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sa / sc; theta[:, 1, 1] = ca / sc * fy; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode='reflection', align_corners=False)
    if torch.rand(1).item() < 0.12:
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x * (1 - msk) + colormap_remap(x) * msk
    gain = 1 + (torch.rand(B, 3, 1, 1, device=d) - 0.5) * 2 * 0.30
    bias = (torch.rand(B, 3, 1, 1, device=d) - 0.5) * 2 * 0.12
    x = x * gain + bias
    x = x + (torch.rand(B, 1, 1, 1, device=d) - 0.5) * 2 * 0.20
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    con = torch.exp((torch.rand(B, 1, 1, 1, device=d) - 0.5) * 2 * 0.35)
    x = (x - m) * con + m
    x = x.clamp(0, 1)
    g = torch.exp((torch.rand(B, 1, 1, 1, device=d) - 0.5) * 2 * 0.35)
    x = x.clamp(1e-4, 1) ** g
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    sat = 0.3 + torch.rand(B, 1, 1, 1, device=d) * 1.1
    x = gray + (x - gray) * sat
    th = (torch.rand(B, device=d) - 0.5) * 2 * math.pi * 0.5
    x = torch.einsum('bij,bjhw->bihw', hue_matrix(th), x).clamp(0, 1)
    if torch.rand(1).item() < 0.06:
        inv = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x * (1 - inv) + (1 - x) * inv
    if torch.rand(1).item() < 0.40:
        sig = 0.5 + torch.rand(1).item() * 1.5
        ks = 7
        k = gauss_kernel(sig, ks, d)
        kk = k.view(1, 1, 1, ks).repeat(3, 1, 1, 1)
        xb = F.conv2d(F.pad(x, (ks // 2, ks // 2, 0, 0), mode='reflect'), kk, groups=3)
        kk2 = k.view(1, 1, ks, 1).repeat(3, 1, 1, 1)
        xb = F.conv2d(F.pad(xb, (0, 0, ks // 2, ks // 2), mode='reflect'), kk2, groups=3)
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x * (1 - msk) + xb * msk
    if torch.rand(1).item() < 0.22:
        f = int(np.random.choice([2, 3, 4]))
        xs = F.interpolate(x, scale_factor=1.0 / f, mode='bilinear', align_corners=False)
        xu = F.interpolate(xs, size=(H, W), mode='bilinear', align_corners=False)
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x * (1 - msk) + xu * msk
    ng = torch.rand(B, 1, 1, 1, device=d) * 0.12
    x = x + torch.randn_like(x) * ng
    spk = torch.rand(B, 1, 1, 1, device=d) * 0.15
    x = x * (1 + torch.randn_like(x) * spk)
    if torch.rand(1).item() < 0.15:
        r = torch.rand(B, 1, H, W, device=d)
        x = torch.where(r < 0.01, torch.zeros_like(x), x)
        x = torch.where(r > 0.99, torch.ones_like(x), x)
    if torch.rand(1).item() < 0.30:
        freq = float(np.random.uniform(0.2, 1.0))
        amp = float(np.random.uniform(0.06, 0.26))
        if np.random.rand() < 0.5:
            line = (torch.sin(torch.arange(H, device=d).float() * freq) * amp).view(1, 1, H, 1)
        else:
            line = (torch.sin(torch.arange(W, device=d).float() * freq) * amp).view(1, 1, 1, W)
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x + line * msk
    x = x.clamp(0, 1)
    if torch.rand(1).item() < 0.20:
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=d), torch.linspace(-1, 1, W, device=d), indexing='ij')
        rad = torch.sqrt(xx ** 2 + yy ** 2)
        rr = float(np.random.uniform(0.9, 1.3))
        vig = torch.clamp(1 - (rad / rr).clamp(0, 1) ** 3, 0, 1).view(1, 1, H, W)
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        x = x * (1 - msk) + x * vig * msk
    if torch.rand(1).item() < 0.30:
        cw = np.random.randint(int(H * 0.1), int(H * 0.35))
        chh = np.random.randint(int(H * 0.1), int(H * 0.35))
        cx = np.random.randint(0, W); cy = np.random.randint(0, H)
        x0 = max(0, cx - cw // 2); x1 = min(W, cx + cw // 2)
        y0 = max(0, cy - chh // 2); y1 = min(H, cy + chh // 2)
        msk = (torch.rand(B, 1, 1, 1, device=d) < 0.5).float()
        patch = torch.rand(B, 3, 1, 1, device=d)
        x[:, :, y0:y1, x0:x1] = x[:, :, y0:y1, x0:x1] * (1 - msk) + patch * msk
    return x.clamp(0, 1)


def normalize(x):
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True) + 1e-4
    return (x - m) / s


def replace_head(m, num_classes):
    if hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif hasattr(m, 'classifier'):
        if isinstance(m.classifier, nn.Sequential):
            for i in range(len(m.classifier) - 1, -1, -1):
                if isinstance(m.classifier[i], nn.Linear):
                    m.classifier[i] = nn.Linear(m.classifier[i].in_features, num_classes)
                    break
        elif isinstance(m.classifier, nn.Linear):
            m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def make_model(arch, num_classes):
    pretrained = True
    try:
        m = getattr(torchvision.models, arch)(weights='IMAGENET1K_V1')
        m = replace_head(m, num_classes)
    except Exception as e:
        print('pretrained unavailable for', arch, '->', repr(e)[:120])
        m = getattr(torchvision.models, arch)(weights=None, num_classes=num_classes)
        pretrained = False
    return m.to(DEVICE), pretrained


def train_one(arch, seed, Xg, yg, samp_w, num_classes, epochs, base_lr, bs):
    torch.manual_seed(seed); np.random.seed(seed)
    model, pretrained = make_model(arch, num_classes)
    lr = base_lr if pretrained else base_lr * 1.6
    ep = epochs if pretrained else int(epochs * 1.25)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    warm = max(1, int(ep * 0.1))
    def lr_lambda(e):
        if e < warm:
            return (e + 1) / warm
        t = (e - warm) / max(1, ep - warm)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = (DEVICE == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    n = Xg.shape[0]
    samp_w_t = torch.from_numpy(samp_w).double()
    idx_all = np.arange(n)
    for e in range(ep):
        model.train()
        perm = torch.multinomial(samp_w_t, n, replacement=True).numpy()
        sel = idx_all[perm]
        for i in range(0, n, bs):
            bidx = sel[i:i + bs]
            bt = torch.from_numpy(bidx).to(DEVICE)
            xb = upscale(Xg[bt].float() / 255.0)
            yb = yg[bt]
            with torch.no_grad():
                xb = normalize(augment(xb))
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.08)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    return model


@torch.no_grad()
def predict_probs(model, Xt, num_classes, bs=256):
    model.eval()
    use_amp = (DEVICE == 'cuda')
    out = []
    for i in range(0, Xt.shape[0], bs):
        xb = normalize(upscale(Xt[i:i + bs].float() / 255.0))
        with torch.cuda.amp.autocast(enabled=use_amp):
            o = F.softmax(model(xb).float(), 1)
            o = o + F.softmax(model(torch.flip(xb, [3])).float(), 1)
            o = o + F.softmax(model(torch.flip(xb, [2])).float(), 1)
            o = o + F.softmax(model(torch.flip(xb, [2, 3])).float(), 1)
        out.append(o.float().cpu())
    return torch.cat(out).numpy()


def main():
    t_start = time.time()
    paths = resolve_paths()
    for k, v in paths.items():
        print(k, '->', v)

    classes = pd.read_csv(paths['classes_csv'])['label'].astype(str).tolist()
    c2i = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    tr = pd.read_csv(paths['train_csv'])
    Xtr = load_npz_images(paths['train_npz'])
    if 'npz_index' in tr.columns:
        Xtr = Xtr[tr['npz_index'].values]
    y = np.array([c2i[str(l)] for l in tr['label']], dtype=np.int64)
    dom = tr['domain_id'].astype(str).values
    modg = tr['modality_group'].astype(str).values

    dom2cls = {d: sorted(set(y[dom == d].tolist())) for d in np.unique(dom)}
    mod2cls = {m: sorted(set(y[modg == m].tolist())) for m in np.unique(modg)}
    dom_major = {d: int(np.bincount(y[dom == d]).argmax()) for d in np.unique(dom)}

    cls_count = np.bincount(y, minlength=num_classes)
    w = 1.0 / np.maximum(cls_count, 1)
    samp_w = w[y].astype(np.float64)

    te = pd.read_csv(paths['test_meta'])
    Xte = load_npz_images(paths['test_npz'])
    if 'npz_index' in te.columns:
        Xte = Xte[te['npz_index'].values]
    test_ids = te['image_id'].astype(str).tolist()
    test_dom = te['domain_id'].astype(str).values
    test_mod = te['modality_group'].astype(str).values if 'modality_group' in te.columns else np.array(['?'] * len(te))

    def candidates(d, mg):
        cand = dom2cls.get(d)
        if not cand:
            cand = mod2cls.get(mg)
        if not cand:
            cand = list(range(num_classes))
        return cand

    baseline_preds = [classes[dom_major.get(test_dom[i], 0)] for i in range(len(test_ids))]
    preds = list(baseline_preds)
    n_models = 0

    try:
        Xg = torch.from_numpy(Xtr).to(DEVICE).permute(0, 3, 1, 2).contiguous()
        yg = torch.from_numpy(y).long().to(DEVICE)
        Xte_g = torch.from_numpy(Xte).to(DEVICE).permute(0, 3, 1, 2).contiguous()

        if DEVICE == 'cuda':
            plan = [('resnet34', 11, 1.5e-3), ('efficientnet_b0', 22, 1.0e-3),
                    ('convnext_tiny', 33, 8.0e-4), ('resnet50', 44, 1.2e-3),
                    ('resnet18', 55, 1.6e-3)]
            epochs = 30
            bs = 64
            budget = 3000.0
        else:
            plan = [('resnet18', 11, 2.0e-3)]
            epochs = 6
            bs = 64
            budget = 1800.0

        probs = np.zeros((len(test_ids), num_classes), dtype=np.float64)
        n_models = 0
        t_train0 = time.time()
        for mi, (arch, seed, lr) in enumerate(plan):
            if mi > 0:
                per_model = (time.time() - t_train0) / mi
                if (time.time() - t_start) + per_model * 1.1 > budget:
                    print('time budget reached; stopping at', mi, 'models')
                    break
            try:
                model = train_one(arch, seed, Xg, yg, samp_w, num_classes, epochs, lr, bs)
                p = predict_probs(model, Xte_g, num_classes)
                if np.isfinite(p).all():
                    probs += p
                    n_models += 1
                    print('model done:', arch, seed, '| n_models=', n_models, '| elapsed', round(time.time() - t_start, 1))
                else:
                    print('non-finite probs; skipping', arch, seed)
                del model
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                print('model failed:', arch, seed, '->', repr(e)[:200])
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

        if n_models > 0:
            new_preds = []
            for i in range(len(test_ids)):
                cand = candidates(test_dom[i], test_mod[i])
                row = probs[i]
                new_preds.append(classes[cand[int(np.argmax(row[cand]))]])
            preds = new_preds
    except Exception as e:
        print('fatal during train/predict; falling back to per-domain baseline ->', repr(e)[:200])
        preds = list(baseline_preds)

    out_dir = 'working'
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = '.'
    out_path = os.path.join(out_dir, 'submission.csv')
    sub = pd.DataFrame({'image_id': test_ids, 'prediction': preds})
    sub.to_csv(out_path, index=False)
    print('wrote', out_path, sub.shape, '| models used', n_models, '| total time', round(time.time() - t_start, 1))


if __name__ == '__main__':
    main()
