#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lab MoE Day 26 (tuan 4 - Green LLM).

Mixture of Experts: tang capacity bang
conditional compute - moi token chi chay qua
k expert trong N expert, nen tong tham so
tang ma FLOPs/token gan nhu khong doi.

Bai A: cai MoELayer (router -> top-k -> cac
expert -> gop co trong so), train Dense vs
MoE tren du lieu toy 4 cum Gaussian 2D.
Bai B: do expert collapse (phan bo token /
expert) roi chua bang aux load-balancing
loss kieu Switch Transformer.
Bai C: top_k=2, vector hoa forward, va
capacity factor (token thua bi roi).

Quy tac thi nghiem (viet truoc khi chay):
thu duy nhat thay doi la lop FFN giua duoc
thay bang MoE; con lai giu nguyen: cung
d_model, cung du lieu, cung optimizer, cung
so epoch, cung lr. MoE co tong tham so lon
hon Dense - do la ban chat cua MoE, khong
phai loi ablation: so sanh cong bang la o
tham so kich hoat / token, khong phai tong
tham so (xem bai1).

Cach chay (venv Day 15, CPU du):
    ~/workspace/venvs/day15/bin/python moe_lab.py
"""

import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# Hang so cho ca lab: doi 1 hang so la doi
# luon thi nghiem, nen moi con so deu co ten.
SEED = 0
N_MOI_CUM = 500
N_CUM = 4
TI_LE_TRAIN = 0.8
DO_LECH_CUM = 0.8
D_VAO = 2
D_MODEL = 32
D_AN_MOE = 16
N_EXPERTS = 4
TOP_K = 1
EPOCHS = 30
LR = 0.01
HE_SO_AUX = 0.01
WARMUP = 2
REPS = 5


def _kiem_tra_so_nguyen(ten, gia_tri, toi_thieu=1):
    """Guard dung chung cho tham so so nguyen.

    Raises:
        ValueError: gia tri la bool, khong phai
            int, hoac nho hon toi_thieu.
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, int) \
            or gia_tri < toi_thieu:
        raise ValueError(
            f"{ten} phai la so nguyen >= {toi_thieu}.")


def _kiem_tra_so_thuc(ten, gia_tri, lon_hon=None,
                      lon_hon_bang=None):
    """Guard dung chung cho tham so so thuc.

    Chi chap nhan int/float (tru bool), huu han.
    Kieu sai (vd chuoi "0.01") phai nem
    ValueError, khong de TypeError tran ra
    tu phep so sanh (checklist 12).

    Raises:
        ValueError: kieu sai, NaN/Inf, hoac
            vuot nguong cho truoc.
    """
    if isinstance(gia_tri, bool) \
            or not isinstance(gia_tri, (int, float)):
        raise ValueError(f"{ten} phai la so.")
    if not math.isfinite(float(gia_tri)):
        raise ValueError(f"{ten} phai huu han.")
    if lon_hon is not None \
            and not float(gia_tri) > lon_hon:
        raise ValueError(f"{ten} phai > {lon_hon}.")
    if lon_hon_bang is not None \
            and not float(gia_tri) >= lon_hon_bang:
        raise ValueError(
            f"{ten} phai >= {lon_hon_bang}.")


def make_clusters(n_moi_cum=N_MOI_CUM, std=DO_LECH_CUM,
                  ti_le_train=TI_LE_TRAIN, seed=SEED):
    """Tao du lieu toy 4 cum Gaussian 2D.

    Tam 4 cum o 4 goc (+-3, +-3), moi cum
    500 diem, nhan = chi so cum. Chia train/
    test 80/20 theo tung cum (stratified) de
    ti le lop giong nhau o ca 2 tap.

    Args:
        n_moi_cum: so diem / cum, int >= 1.
        std: do lech chuan moi cum, so > 0.
        ti_le_train: ti le train, so trong
            (0, 1).
        seed: seed de chia train/test lap lai
            duoc, int >= 0.

    Returns:
        (X_train, y_train, X_test, y_test):
        X float32 (N, 2), y long (N,).

    Raises:
        ValueError: tham so sai kieu / pham vi.
    """
    _kiem_tra_so_nguyen("n_moi_cum", n_moi_cum)
    _kiem_tra_so_thuc("std", std, lon_hon=0)
    _kiem_tra_so_thuc("ti_le_train", ti_le_train)
    if not 0.0 < float(ti_le_train) < 1.0:
        raise ValueError(
            "ti_le_train phai nam trong (0, 1).")
    if isinstance(seed, bool) \
            or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed phai la so nguyen >= 0.")
    gen = torch.Generator()
    gen.manual_seed(seed)
    tam = torch.tensor([[-3.0, -3.0], [-3.0, 3.0],
                        [3.0, -3.0], [3.0, 3.0]])
    n_train_cum = int(n_moi_cum * ti_le_train)
    xs_train, ys_train, xs_test, ys_test = [], [], [], []
    for cum in range(N_CUM):
        diem = tam[cum] + std * torch.randn(
            n_moi_cum, 2, generator=gen)
        nhan = torch.full((n_moi_cum,), cum,
                          dtype=torch.long)
        thu_tu = torch.randperm(n_moi_cum,
                                generator=gen)
        diem, nhan = diem[thu_tu], nhan[thu_tu]
        xs_train.append(diem[:n_train_cum])
        ys_train.append(nhan[:n_train_cum])
        xs_test.append(diem[n_train_cum:])
        ys_test.append(nhan[n_train_cum:])
    x_train = torch.cat(xs_train, dim=0)
    y_train = torch.cat(ys_train, dim=0)
    x_test = torch.cat(xs_test, dim=0)
    y_test = torch.cat(ys_test, dim=0)
    # Tron lan nua de thu tu cum khong anh
    # huong train (van dung generator co seed).
    thu_tu = torch.randperm(len(x_train), generator=gen)
    return (x_train[thu_tu], y_train[thu_tu],
            x_test, y_test)


def dem_tham_so(model):
    """Dem tong so tham so cua mot model.

    Raises:
        ValueError: model khong phai nn.Module.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    return sum(p.numel() for p in model.parameters())


def _kiem_tra_tensor_2d(ten, x, d_in=None):
    """Guard dung chung cho tensor 2D float.

    Kiem tra: la tensor, 2 chieu, dtype so
    thuc, huu han, khong rong, va (neu co
    d_in) chieu cuoi bang d_in.

    Raises:
        ValueError: bat ky dieu kien nao sai.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{ten} phai la tensor.")
    if x.dim() != 2:
        raise ValueError(f"{ten} phai co 2 chieu.")
    if not x.is_floating_point():
        raise ValueError(f"{ten} phai la so thuc.")
    if x.size(0) == 0 or x.size(1) == 0:
        raise ValueError(f"{ten} khong duoc rong.")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{ten} phai huu han.")
    if d_in is not None and x.size(1) != d_in:
        raise ValueError(
            f"{ten} phai co chieu cuoi = {d_in}.")


class MoELayer(nn.Module):
    """Lop MoE: router + top-k + N expert MLP.

    Moi expert la MLP nho d_in -> d_hidden ->
    d_in. Router la Linear(d_in, n_experts),
    softmax ra gating, chi top-k expert duoc
    chay, dau ra gop co trong so (trong so
    top-k da chuan hoa ve tong = 1).

    Args:
        d_in: chieu vao / ra, int >= 1.
        d_hidden: chieu an cua moi expert,
            int >= 1.
        n_experts: so expert, int >= 1.
        top_k: so expert chay / token, int
            trong [1, n_experts].
        capacity_factor: None = khong gioi
            han; so > 0 = moi expert chi nhan
            toi da capacity_factor * (so token
            / n_experts) token / batch, token
            thua bi roi (dong gop = 0).
    """

    def __init__(self, d_in, d_hidden, n_experts=4,
                 top_k=1, capacity_factor=None):
        super().__init__()
        _kiem_tra_so_nguyen("d_in", d_in)
        _kiem_tra_so_nguyen("d_hidden", d_hidden)
        _kiem_tra_so_nguyen("n_experts", n_experts)
        _kiem_tra_so_nguyen("top_k", top_k)
        if top_k > n_experts:
            raise ValueError(
                "top_k khong duoc lon hon n_experts.")
        if capacity_factor is not None:
            _kiem_tra_so_thuc("capacity_factor",
                              capacity_factor,
                              lon_hon=0)
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_in, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.ReLU(),
                nn.Linear(d_hidden, d_in))
            for _ in range(n_experts)
        ])

    def _tinh_gating(self, x):
        """Tinh gating + top-k da chuan hoa.

        Returns:
            (trong_so, topk_idx): trong_so
            (batch, top_k), tong moi hang = 1;
            topk_idx (batch, top_k) long.
        """
        logits = self.router(x)
        gating = torch.softmax(logits, dim=-1)
        vals, idx = gating.topk(self.top_k, dim=-1)
        # Chuan hoa lai: top_k=2 thi tong 2
        # trong so softmax co the chi con 0.6,
        # khong chia lai thi output bi co nho
        # am tham (bay 2 trong de bai).
        trong_so = vals / vals.sum(dim=-1,
                                   keepdim=True)
        return trong_so, idx, gating

    def forward(self, x):
        """Chay MoE, tra (out, router_probs).

        Ban "cham ma ro": vong lap theo tung
        token, cong tung expert duoc chon.
        Trong so gating luon duoc nhan vao -
        neu cong thang expert ma quen nhan
        trong so thi router khong nhan duoc
        gradient (bay 1 trong de bai).

        Args:
            x: tensor float (batch, d_in),
                huu han, khong rong.

        Returns:
            (out, router_probs): out cung
            shape voi x; router_probs la gating
            day du (batch, n_experts), truoc
            khi top-k, tong moi hang = 1.

        Raises:
            ValueError: x sai dtype / shape /
                chua NaN-Inf / rong.
        """
        _kiem_tra_tensor_2d("x", x, d_in=self.d_in)
        trong_so, topk_idx, gating = self._tinh_gating(x)
        # zeros_like(x) lay dung device cua x
        # (checklist 3): khong dung torch.zeros
        # mac dinh CPU khi x o GPU.
        out = torch.zeros_like(x)
        if self.capacity_factor is None:
            for i in range(x.size(0)):
                for k in range(self.top_k):
                    chuyen_gia = topk_idx[i, k].item()
                    out[i] = out[i] + trong_so[i, k] \
                        * self.experts[chuyen_gia](x[i])
        else:
            suc_chua = max(
                1, int(self.capacity_factor
                       * x.size(0) / self.n_experts))
            da_nhan = [0] * self.n_experts
            for i in range(x.size(0)):
                for k in range(self.top_k):
                    chuyen_gia = topk_idx[i, k].item()
                    if da_nhan[chuyen_gia] >= suc_chua:
                        continue
                    da_nhan[chuyen_gia] += 1
                    out[i] = out[i] + trong_so[i, k] \
                        * self.experts[chuyen_gia](x[i])
        return out, gating


def moe_forward_vectorized(layer, x):
    """Forward MoE vector hoa (Bai C2).

    Cung dap an voi MoELayer.forward, nhung
    gom token theo expert roi chay expert
    theo batch (index_select + index_add),
    khong lap tung token. Chi dung khi
    capacity_factor=None (ban nay khong cai
    roi token).

    Args:
        layer: MoELayer da khoi tao.
        x: tensor float (batch, d_in) nhu
            forward.

    Returns:
        (out, router_probs) giong forward.

    Raises:
        ValueError: layer khong phai MoELayer,
            x sai nhu forward, hoac layer co
            capacity_factor (khong ho tro).
    """
    if not isinstance(layer, MoELayer):
        raise ValueError("layer phai la MoELayer.")
    if layer.capacity_factor is not None:
        raise ValueError(
            "ban vector hoa khong ho tro capacity.")
    _kiem_tra_tensor_2d("x", x, d_in=layer.d_in)
    trong_so, topk_idx, gating = layer._tinh_gating(x)
    batch, top_k = topk_idx.shape
    # Chi so tren dung device cua x.
    chi_token = torch.arange(
        batch, device=x.device).repeat_interleave(top_k)
    chi_expert = topk_idx.reshape(-1)
    trong_so_phang = trong_so.reshape(-1)
    out = torch.zeros_like(x)
    for e in range(layer.n_experts):
        mask = chi_expert == e
        if not bool(mask.any()):
            continue
        tok = chi_token[mask]
        w = trong_so_phang[mask].unsqueeze(1)
        dong_gop = layer.experts[e](x[tok]) * w
        out = out.index_add(0, tok, dong_gop)
    return out, gating


def _kiem_tra_router_probs_topk(router_probs, topk_idx,
                                n_experts):
    """Guard dung chung cho aux loss / phan bo.

    Raises:
        ValueError: router_probs khong phai
            tensor float 2D huu han, hang
            khong cong ve 1, so cot khac
            n_experts; topk_idx khong phai
            tensor long 2D, cung batch, gia
            tri ngoai [0, n_experts).
    """
    _kiem_tra_tensor_2d("router_probs", router_probs)
    if router_probs.size(1) != n_experts:
        raise ValueError(
            "router_probs phai co n_experts cot.")
    tong_hang = router_probs.sum(dim=-1)
    if not bool(torch.allclose(
            tong_hang, torch.ones_like(tong_hang),
            atol=1e-5)):
        raise ValueError(
            "moi hang router_probs phai cong ve 1.")
    if not isinstance(topk_idx, torch.Tensor):
        raise ValueError("topk_idx phai la tensor.")
    if topk_idx.dim() != 2 \
            or topk_idx.size(0) != router_probs.size(0):
        raise ValueError(
            "topk_idx phai 2D va cung batch voi "
            "router_probs.")
    if topk_idx.dtype != torch.long:
        raise ValueError("topk_idx phai la long.")
    if topk_idx.numel() == 0:
        raise ValueError("topk_idx khong duoc rong.")
    if int(topk_idx.min()) < 0 \
            or int(topk_idx.max()) >= n_experts:
        raise ValueError(
            "topk_idx vuot ngoai [0, n_experts).")


def aux_load_balancing_loss(router_probs, topk_idx,
                            n_experts):
    """Aux loss can bang tai kieu Switch.

    f_i: ti le token thuc su gui toi expert i
    (mean cua one-hot top-k tren ca batch va
    ca top_k, tong = 1). P_i: xac suat trung
    binh router gan cho expert i (mean cua
    router_probs theo batch). aux = n * sum_i
    (f_i * P_i): can bang thi f_i = P_i = 1/n
    nen aux = 1; don vao 1 expert thi aux ~ n.

    Args:
        router_probs: tensor float (batch,
            n_experts), hang cong ve 1.
        topk_idx: tensor long (batch, top_k).
        n_experts: int >= 1.

    Returns:
        Tensor vo huong (0-d): gia tri aux loss.

    Raises:
        ValueError: dau vao sai nhu helper
            _kiem_tra_router_probs_topk.
    """
    _kiem_tra_so_nguyen("n_experts", n_experts)
    _kiem_tra_router_probs_topk(router_probs, topk_idx,
                                n_experts)
    one_hot = F.one_hot(topk_idx,
                        num_classes=n_experts).float()
    f_i = one_hot.mean(dim=(0, 1))
    p_i = router_probs.mean(dim=0)
    return n_experts * (f_i * p_i).sum()


def phan_bo_expert(router_probs, topk_idx, n_experts):
    """Ti le % token moi expert nhan (tren tap).

    Dem tren moi luot gui (batch * top_k),
    chia cho tong luot gui, nhan 100 - tong
    cac % = 100. Tap con rong khong xay ra
    vi batch >= 1 da duoc guard.

    Returns:
        List float, do dai n_experts, tong 100.

    Raises:
        ValueError: dau vao sai nhu helper
            _kiem_tra_router_probs_topk.
    """
    _kiem_tra_so_nguyen("n_experts", n_experts)
    _kiem_tra_router_probs_topk(router_probs, topk_idx,
                                n_experts)
    dem = torch.bincount(topk_idx.reshape(-1),
                         minlength=n_experts)
    tong = float(topk_idx.numel())
    return [float(c) / tong * 100.0 for c in dem]

class DenseModel(nn.Module):
    """MLP dense doi chieu: 2->32->32->4.

    forward tra (logits, None) de chung giao
    dien voi MoEModel: None nghia la khong co
    router_probs, train() se khong tinh aux.
    """

    def __init__(self, d_vao=D_VAO, d_model=D_MODEL,
                 n_lop=4):
        super().__init__()
        _kiem_tra_so_nguyen("d_vao", d_vao)
        _kiem_tra_so_nguyen("d_model", d_model)
        _kiem_tra_so_nguyen("n_lop", n_lop)
        self.fc1 = nn.Linear(d_vao, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, n_lop)

    def forward(self, x):
        _kiem_tra_tensor_2d("x", x, d_in=D_VAO)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h), None


class MoEModel(nn.Module):
    """MLP MoE: 2->32->MoE(32,16)->4.

    forward tra (logits, router_probs) de
    train() tinh duoc aux loss va do phan bo.
    """

    def __init__(self, d_vao=D_VAO, d_model=D_MODEL,
                 d_an=D_AN_MOE, n_experts=N_EXPERTS,
                 top_k=TOP_K, capacity_factor=None,
                 n_lop=4):
        super().__init__()
        _kiem_tra_so_nguyen("d_vao", d_vao)
        _kiem_tra_so_nguyen("d_model", d_model)
        _kiem_tra_so_nguyen("d_an", d_an)
        _kiem_tra_so_nguyen("n_lop", n_lop)
        self.fc1 = nn.Linear(d_vao, d_model)
        self.moe = MoELayer(d_model, d_an,
                            n_experts=n_experts,
                            top_k=top_k,
                            capacity_factor=capacity_factor)
        self.fc3 = nn.Linear(d_model, n_lop)

    def forward(self, x):
        _kiem_tra_tensor_2d("x", x, d_in=D_VAO)
        h = F.relu(self.fc1(x))
        h, router_probs = self.moe(h)
        return self.fc3(h), router_probs


def tham_so_kich_hoat_moe(model):
    """Tham so kich hoat / token cua MoEModel.

    Tong tru di (n_experts - 1) expert dang
    ngu: moi token chi chay top_k expert, nen
    voi top_k=1 chi 1 expert duoc tinh. Router
    va cac Linear ngoai MoE luon chay nen van
    tinh du. Cong thuc dung rieng 1 expert
    lam chuan (ca expert cung kich thuoc).

    Returns:
        (tong, kich_hoat): 2 so nguyen.

    Raises:
        ValueError: model khong phai MoEModel.
    """
    if not isinstance(model, MoEModel):
        raise ValueError("model phai la MoEModel.")
    tong = dem_tham_so(model)
    mot_expert = dem_tham_so(model.moe.experts[0])
    kich_hoat = tong - (model.moe.n_experts
                        - model.moe.top_k) * mot_expert
    return tong, kich_hoat


def _kiem_tra_du_lieu_huan_luyen(x_train, y_train,
                                 x_test, y_test):
    """Guard X float32 huu han, y long, khop batch."""
    for ten, x in (("x_train", x_train),
                   ("x_test", x_test)):
        _kiem_tra_tensor_2d(ten, x, d_in=D_VAO)
        if x.dtype != torch.float32:
            raise ValueError(f"{ten} phai float32.")
    for ten, y, x in (("y_train", y_train, x_train),
                      ("y_test", y_test, x_test)):
        if not isinstance(y, torch.Tensor):
            raise ValueError(f"{ten} phai la tensor.")
        if y.dim() != 1 or y.size(0) != x.size(0):
            raise ValueError(
                f"{ten} phai 1D va cung batch voi x.")
        if y.dtype != torch.long:
            raise ValueError(f"{ten} phai la long.")
        if y.numel() == 0:
            raise ValueError(f"{ten} khong duoc rong.")
        if int(y.min()) < 0 \
                or int(y.max()) >= N_CUM:
            raise ValueError(
                f"{ten} vuot ngoai [0, {N_CUM}).")


def danh_gia(model, x, y):
    """Accuracy tren 1 tap (0..1), khong train.

    Raises:
        ValueError: model khong phai nn.Module,
            du lieu sai nhu helper tren.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    _kiem_tra_du_lieu_huan_luyen(x, y, x, y)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(
                "logits phai huu han khi danh gia.")
        du_doan = logits.argmax(dim=-1)
        return float((du_doan == y).float().mean())


def train_model(model, x_train, y_train, x_test, y_test,
                epochs=EPOCHS, lr=LR, he_so_aux=0.0):
    """Train full-batch, tra accuracy + loss cuoi.

    Optimizer Adam, loss CrossEntropy. Neu
    model co router_probs va he_so_aux > 0 thi
    total = task + he_so_aux * aux; neu khong
    thi total = task (aux tra None, khong phai
    0.0 - 0.0 se noi doi la "khong co aux").

    Args:
        model: DenseModel hoac MoEModel.
        epochs: int >= 1. lr: so > 0.
        he_so_aux: so >= 0.

    Returns:
        {"accuracy": float 0..1, "task_loss": float,
         "aux_loss": float hoac None}.

    Raises:
        ValueError: model/du lieu/tham so sai.
    """
    if not isinstance(model, nn.Module):
        raise ValueError("model phai la nn.Module.")
    _kiem_tra_du_lieu_huan_luyen(x_train, y_train,
                                 x_test, y_test)
    _kiem_tra_so_nguyen("epochs", epochs)
    _kiem_tra_so_thuc("lr", lr, lon_hon=0)
    _kiem_tra_so_thuc("he_so_aux", he_so_aux,
                      lon_hon_bang=0)
    toi_uu = torch.optim.Adam(model.parameters(), lr=lr)
    task_cuoi, aux_cuoi = float("nan"), None
    top_k = model.moe.top_k if hasattr(model, "moe") \
        else TOP_K
    for _ in range(epochs):
        model.train()
        toi_uu.zero_grad()
        logits, router_probs = model(x_train)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError(
                "logits train phai huu han.")
        task_cuoi = F.cross_entropy(logits, y_train)
        tong_loss = task_cuoi
        if router_probs is not None and he_so_aux > 0:
            topk_idx = router_probs.topk(
                top_k, dim=-1).indices
            aux_cuoi = aux_load_balancing_loss(
                router_probs, topk_idx,
                router_probs.size(1))
            tong_loss = task_cuoi + he_so_aux * aux_cuoi
        tong_loss.backward()
        toi_uu.step()
    acc = danh_gia(model, x_test, y_test)
    return {"accuracy": acc,
            "task_loss": float(task_cuoi),
            "aux_loss": None if aux_cuoi is None
            else float(aux_cuoi)}


def lay_router_probs(model, x):
    """Lay (router_probs, topk_idx) tren 1 tap.

    Dung de do phan bo expert sau khi train.
    Tra topk_idx theo top_k cua lop MoE.

    Raises:
        ValueError: model khong phai MoEModel.
    """
    if not isinstance(model, MoEModel):
        raise ValueError("model phai la MoEModel.")
    _kiem_tra_tensor_2d("x", x, d_in=D_VAO)
    model.eval()
    with torch.no_grad():
        _, router_probs = model(x)
    topk_idx = router_probs.topk(
        model.moe.top_k, dim=-1).indices
    return router_probs, topk_idx


def bench(fn, warmup=WARMUP, reps=REPS):
    """Do thoi gian forward: TRUNG VI ms / lan.

    Dung time.perf_counter() (checklist 5),
    duoi no_grad. Tren GPU dong bo CUDA sau
    warmup va sau moi lan chay (checklist 21);
    tren CPU nhanh dong bo khong chay.

    Returns:
        Trung vi thoi gian, mili-giay / lan.
    """
    _kiem_tra_so_nguyen("warmup", warmup, toi_thieu=0)
    _kiem_tra_so_nguyen("reps", reps)
    if not callable(fn):
        raise ValueError("fn phai goi duoc.")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        thoi_gian = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            thoi_gian.append(time.perf_counter() - t0)
    return float(statistics.median(thoi_gian) * 1000.0)


def _expect_value_error(fn, mo_ta):
    """Chay fn, doi no nem ValueError, in ket qua."""
    try:
        fn()
    except ValueError as exc:
        print(f"  guard {mo_ta}: chan dung ({exc})")
        return True
    print(f"  CANH BAO: guard {mo_ta} KHONG chan!")
    return False


def _sai_so_lon_nhat(a, b):
    """Sai so tuyet doi lon nhat giua 2 tensor.

    Detach truoc khi doi sang float: output
    cua lop MoE mang requires_grad=True (do
    tham so lop), float() truc tiep se sinh
    UserWarning va co the anh huong autograd.
    """
    return float((a.detach() - b.detach()).abs().max())


def tu_kiem_tra():
    """Tu kiem tra bang assert, chay truoc lab.

    Kiem tra guard, MoE forward dung (so voi
    tinh tay), trong so top-k chuan hoa ve 1,
    gradient chay toi router, aux loss o 2
    cuc (can bang ~1, collapse ~n), vector
    hoa bang ban vong lap (ca gradient),
    capacity roi dung token, va du lieu toy
    dung kich thuoc / ti le lop. Chi IN khi
    assert da qua (checklist 13).
    """
    torch.manual_seed(123)
    ket_qua = [
        _expect_value_error(
            lambda: MoELayer(32, 16, top_k=0),
            "top_k=0"),
        _expect_value_error(
            lambda: MoELayer(32, 16, n_experts=4,
                             top_k=5),
            "top_k>n_experts"),
        _expect_value_error(
            lambda: MoELayer(0, 16), "d_in=0"),
        _expect_value_error(
            lambda: MoELayer(32, 16,
                             capacity_factor=0),
            "capacity_factor=0"),
        _expect_value_error(
            lambda: MoELayer(32, 16,
                             capacity_factor="1.25"),
            "capacity_factor dang chuoi"),
        _expect_value_error(
            lambda: MoELayer(32, 16).forward(
                torch.zeros(4, 32, dtype=torch.long)),
            "x long"),
        _expect_value_error(
            lambda: MoELayer(32, 16).forward(
                torch.full((4, 32), float("nan"))),
            "x chua NaN"),
        _expect_value_error(
            lambda: MoELayer(32, 16).forward(
                torch.zeros(32)), "x 1 chieu"),
        _expect_value_error(
            lambda: MoELayer(32, 16).forward(
                torch.zeros(0, 32)), "x rong"),
        _expect_value_error(
            lambda: aux_load_balancing_loss(
                torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                torch.tensor([[0]]), 4),
            "router_probs hang khong cong ve 1"),
        _expect_value_error(
            lambda: aux_load_balancing_loss(
                torch.full((2, 4), 0.25),
                torch.tensor([[4], [0]]), 4),
            "topk_idx vuot pham vi"),
        _expect_value_error(
            lambda: make_clusters(n_moi_cum=0),
            "make_clusters n=0"),
        _expect_value_error(
            lambda: train_model(
                DenseModel(), torch.zeros(4, 2),
                torch.zeros(4, dtype=torch.long),
                torch.zeros(4, 2),
                torch.zeros(4, dtype=torch.long),
                lr=-0.01),
            "train lr am"),
        _expect_value_error(
            lambda: bench(42), "bench fn khong goi"),
        _expect_value_error(
            lambda: moe_forward_vectorized(
                MoELayer(32, 16, capacity_factor=1.25),
                torch.randn(4, 32)),
            "vector hoa co capacity"),
    ]
    assert all(ket_qua), "Co guard khong chan dung!"

    # Du lieu toy: 2000 diem, train 1600 / test
    # 400, moi lop 400 train / 100 test.
    x_tr, y_tr, x_te, y_te = make_clusters()
    assert x_tr.shape == (1600, 2)
    assert x_te.shape == (400, 2)
    assert y_tr.shape == (1600,)
    assert all(int((y_tr == c).sum()) == 400
               for c in range(4))
    assert all(int((y_te == c).sum()) == 100
               for c in range(4))

    # MoE forward: shape, gating cong ve 1.
    torch.manual_seed(7)
    lop = MoELayer(32, 16, n_experts=4, top_k=2)
    x = torch.randn(8, 32)
    out, probs = lop(x)
    assert out.shape == (8, 32)
    assert probs.shape == (8, 4)
    assert bool(torch.allclose(
        probs.sum(dim=-1), torch.ones(8), atol=1e-6))

    # Trong so top-k chuan hoa: tinh tay bang
    # gating.topk + chia tong, so voi forward.
    vals, idx = probs.topk(2, dim=-1)
    w = vals / vals.sum(dim=-1, keepdim=True)
    out_tay = torch.zeros_like(x)
    for i in range(8):
        for k in range(2):
            out_tay[i] = out_tay[i] + w[i, k] \
                * lop.experts[idx[i, k].item()](x[i])
    assert _sai_so_lon_nhat(out, out_tay) <= 1e-6, \
        "forward phai bang tinh tay co trong so."

    # Gradient chay toi router (bay 1: quen
    # nhan trong so thi grad router = 0).
    lop2 = MoELayer(8, 4, n_experts=2, top_k=1)
    x2 = torch.randn(4, 8, requires_grad=True)
    out2, _ = lop2(x2)
    out2.sum().backward()
    assert lop2.router.weight.grad is not None
    assert float(lop2.router.weight.grad.abs().sum()) \
        > 0, "router phai nhan gradient."

    # Aux loss 2 cuc: can bang ~1, collapse ~4.
    probs_deu = torch.full((8, 4), 0.25)
    idx_deu = torch.tensor([[0], [1], [2], [3],
                            [0], [1], [2], [3]])
    aux_deu = aux_load_balancing_loss(
        probs_deu, idx_deu, 4)
    assert abs(float(aux_deu) - 1.0) <= 1e-6, \
        "can bang thi aux phai ~1."
    probs_sup = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 8)
    idx_sup = torch.zeros(8, 1, dtype=torch.long)
    aux_sup = aux_load_balancing_loss(
        probs_sup, idx_sup, 4)
    assert abs(float(aux_sup) - 4.0) <= 1e-6, \
        "collapse thi aux phai ~n_experts."
    # Phan bo: collapse thi [100, 0, 0, 0].
    pb_sup = phan_bo_expert(probs_sup, idx_sup, 4)
    assert pb_sup == [100.0, 0.0, 0.0, 0.0]
    assert abs(sum(pb_sup) - 100.0) <= 1e-9

    # Vector hoa bang ban vong lap, ca gradient.
    torch.manual_seed(3)
    lop3 = MoELayer(16, 8, n_experts=4, top_k=2)
    x3 = torch.randn(6, 16)
    out_vong, _ = lop3(x3)
    out_vec, _ = moe_forward_vectorized(lop3, x3)
    assert _sai_so_lon_nhat(out_vong, out_vec) \
        <= 1e-6, "vector hoa phai bang vong lap."
    x3g = torch.randn(6, 16, requires_grad=True)
    out_vong_g, _ = lop3(x3g)
    out_vong_g.sum().backward()
    grad_vong = lop3.router.weight.grad.clone()
    lop3.router.weight.grad = None
    out_vec_g, _ = moe_forward_vectorized(lop3, x3g)
    out_vec_g.sum().backward()
    assert _sai_so_lon_nhat(
        grad_vong, lop3.router.weight.grad) <= 1e-6, \
        "gradient vector hoa phai bang vong lap."

    # Capacity: ep router don het ve expert 0,
    # capacity=1.0, batch 8, 4 expert -> moi
    # expert toi da 2 token. Chi 2 token dau
    # co output khac 0, 6 token sau bi roi.
    lop4 = MoELayer(4, 4, n_experts=4, top_k=1,
                    capacity_factor=1.0)
    with torch.no_grad():
        lop4.router.weight.zero_()
        lop4.router.bias.copy_(
            torch.tensor([10.0, -10.0, -10.0, -10.0]))
    x4 = torch.randn(8, 4)
    out4, _ = lop4(x4)
    assert bool((out4[:2].abs().sum(dim=-1) > 0).all())
    assert bool((out4[2:].abs().sum(dim=-1) == 0).all()), \
        "token vuot capacity phai bi roi (out=0)."
    print("Tu kiem tra: PASS (tat ca assert deu qua).")

def _dinh_dang_phan_bo(phan_bo):
    """In phan bo dang 'Expert 0: 82% | ...'."""
    return " | ".join(
        f"Expert {i}: {p:.0f}%"
        for i, p in enumerate(phan_bo))


def _ket_luan_ti_le(ti_le, ten_nhanh, ten_cham,
                    nguong_duoi=0.8, nguong_tren=1.25):
    """Ket luan ti le toc do 3 nhanh (checklist 22).

    < nguong_duoi: nhanh hon; trong khoang:
    gan nhu hoa; > nguong_tren: cham hon.
    Khong dung else mac dinh "cham hon X"
    vi se in nguoc khi ti le < 1.
    """
    if ti_le < nguong_duoi:
        return (f"{ten_nhanh} nhanh hon "
                f"({ti_le:.2f}x thoi gian)")
    if ti_le <= nguong_tren:
        return (f"{ten_nhanh} gan nhu hoa voi "
                f"{ten_cham} ({ti_le:.2f}x)")
    return (f"{ten_nhanh} cham hon {ten_cham} "
            f"{ti_le:.2f}x")


def bai1(x_train, y_train, x_test, y_test):
    """Bai A: Dense vs MoE (top_k=1), 30 epoch.

    Thu duy nhat thay doi la lop FFN giua:
    Dense dung Linear(32->32), MoE dung
    MoELayer(32, 16, 4 expert, top_k=1). Cung
    du lieu, cung optimizer Adam lr=0.01, cung
    30 epoch. Seed dat lai truoc moi model de
    khoi tao dau vao giong nhau toi da co the.
    """
    print("\n" + "=" * 72)
    print("BAI A - Dense vs MoE (top_k=1)")
    print("=" * 72)
    print("Thu duy nhat thay doi la lop FFN giua "
          "duoc thay bang MoE; cung d_model, cung "
          "du lieu, cung optimizer, cung so epoch.")
    torch.manual_seed(SEED)
    dense = DenseModel()
    kq_dense = train_model(dense, x_train, y_train,
                           x_test, y_test)
    torch.manual_seed(SEED)
    moe = MoEModel(top_k=TOP_K)
    kq_moe = train_model(moe, x_train, y_train,
                         x_test, y_test)
    ts_dense = dem_tham_so(dense)
    ts_moe, ts_kich_hoat = tham_so_kich_hoat_moe(moe)
    print(f"  Dense: tham so={ts_dense} | "
          f"accuracy test={kq_dense['accuracy']:.3f}")
    print(f"  MoE: tong tham so={ts_moe} "
          f"({ts_moe / ts_dense:.2f}x Dense) | "
          f"kich hoat/token={ts_kich_hoat} "
          f"({ts_kich_hoat / ts_dense:.2f}x Dense) | "
          f"accuracy test={kq_moe['accuracy']:.3f}")
    chenh = kq_moe["accuracy"] - kq_dense["accuracy"]
    n_mau = round(abs(chenh) * len(y_test))
    if abs(chenh) < 0.005:
        ket = (f"gan nhu hoa (chenh "
               f"{chenh * 100:.1f} diem %, ~{n_mau} "
               f"mau test) - MoE khong hon Dense "
               f"tren du lieu toy de nay")
    elif chenh > 0:
        ket = (f"MoE cao hon {chenh * 100:.1f} diem "
               f"% (~{n_mau} mau test)")
    else:
        ket = (f"MoE thap hon {-chenh * 100:.1f} "
               f"diem % (~{n_mau} mau test)")
    print(f"Ket luan: {ket}. Tong tham so MoE gap "
          f"{ts_moe / ts_dense:.2f}x nhung tham so "
          f"kich hoat/token chi "
          f"{ts_kich_hoat / ts_dense:.2f}x - dung "
          f"ban chat conditional compute.")
    return {"dense_acc": kq_dense["accuracy"],
            "moe_acc": kq_moe["accuracy"],
            "ts_dense": ts_dense, "ts_moe": ts_moe,
            "ts_kich_hoat": ts_kich_hoat,
            "moe_model": moe}


def bai2(moe_truoc, x_train, y_train, x_test, y_test):
    """Bai B: collapse + aux loss (he so 0.01).

    Do phan bo tren tap test cua model MoE
    Bai A (khong aux), roi train lai MoE tu
    dau chi khac he_so_aux (0 -> 0.01, checklist
    10: diff 2 loi goi train chi co 1 bien).
    """
    print("\n" + "=" * 72)
    print("BAI B - Expert collapse + aux loss")
    print("=" * 72)
    probs_truoc, idx_truoc = lay_router_probs(
        moe_truoc, x_test)
    pb_truoc = phan_bo_expert(
        probs_truoc, idx_truoc, N_EXPERTS)
    print(f"  Truoc aux: {_dinh_dang_phan_bo(pb_truoc)}")
    torch.manual_seed(SEED)
    moe_aux = MoEModel(top_k=TOP_K)
    kq_aux = train_model(moe_aux, x_train, y_train,
                         x_test, y_test,
                         he_so_aux=HE_SO_AUX)
    probs_sau, idx_sau = lay_router_probs(moe_aux, x_test)
    pb_sau = phan_bo_expert(probs_sau, idx_sau, N_EXPERTS)
    print(f"  Sau aux (he so {HE_SO_AUX}): "
          f"{_dinh_dang_phan_bo(pb_sau)}")
    print(f"  Accuracy test: truoc aux "
          f"{danh_gia(moe_truoc, x_test, y_test):.3f} "
          f"-> sau aux {kq_aux['accuracy']:.3f} "
          f"(aux loss cuoi={kq_aux['aux_loss']:.3f})")
    lon_truoc, lon_sau = max(pb_truoc), max(pb_sau)
    chenh_acc = kq_aux["accuracy"] - danh_gia(
        moe_truoc, x_test, y_test)
    n_mau = round(abs(chenh_acc) * len(y_test))
    if lon_sau < lon_truoc - 5:
        deu = (f"deu hon ro: expert nhieu nhat tu "
               f"{lon_truoc:.0f}% xuong {lon_sau:.0f}%")
    elif lon_sau < lon_truoc:
        deu = (f"deu hon nhe: expert nhieu nhat tu "
               f"{lon_truoc:.0f}% xuong {lon_sau:.0f}%")
    elif lon_sau > lon_truoc:
        deu = (f"KHONG deu hon: expert nhieu nhat "
               f"tu {lon_truoc:.0f}% len {lon_sau:.0f}%")
    else:
        deu = "phan bo khong doi"
    if abs(chenh_acc) < 0.005:
        acc = (f"accuracy gan nhu giu nguyen "
               f"({chenh_acc * 100:.1f} diem %, "
               f"~{n_mau} mau)")
    elif chenh_acc > 0:
        acc = (f"accuracy tang {chenh_acc * 100:.1f} "
               f"diem % (~{n_mau} mau)")
    else:
        acc = (f"accuracy giam {-chenh_acc * 100:.1f} "
               f"diem % (~{n_mau} mau) - chap nhan "
               f"duoc vi aux ep router chia deu, "
               f"doi lai moi expert deu duoc hoc")
    print(f"Ket luan: {deu}; {acc}.")
    return {"pb_truoc": pb_truoc, "pb_sau": pb_sau,
            "acc_sau": kq_aux["accuracy"],
            "moe_aux_model": moe_aux}


def bai3(x_train, y_train, x_test, y_test, acc_moe_k1):
    """Bai C: top_k=2, vector hoa, capacity.

    Moi muc chi doi dung 1 thu so voi Bai A:
    C1 chi doi top_k 1->2; C2 chi doi cach
    chay forward; C3 chi doi capacity_factor.
    """
    print("\n" + "=" * 72)
    print("BAI C - Nang cao: top_k=2, vector hoa, "
          "capacity")
    print("=" * 72)
    # C1: top_k=2, khong aux (giong Bai A).
    torch.manual_seed(SEED)
    moe_k2 = MoEModel(top_k=2)
    kq_k2 = train_model(moe_k2, x_train, y_train,
                        x_test, y_test)
    probs_k2, idx_k2 = lay_router_probs(moe_k2, x_test)
    pb_k2 = phan_bo_expert(probs_k2, idx_k2, N_EXPERTS)
    chenh = kq_k2["accuracy"] - acc_moe_k1
    n_mau = round(abs(chenh) * len(y_test))
    if abs(chenh) < 0.005:
        acc_k2 = (f"accuracy gan nhu hoa "
                  f"({chenh * 100:.1f} diem %, "
                  f"~{n_mau} mau)")
    elif chenh > 0:
        acc_k2 = (f"accuracy tang {chenh * 100:.1f} "
                  f"diem % (~{n_mau} mau)")
    else:
        acc_k2 = (f"accuracy giam {-chenh * 100:.1f} "
                  f"diem % (~{n_mau} mau)")
    print(f"  C1 top_k=2: accuracy="
          f"{kq_k2['accuracy']:.3f} | "
          f"{_dinh_dang_phan_bo(pb_k2)}")
    print(f"     {acc_k2}. FLOPs expert/token tang "
          f"~2x so voi top_k=1 (2 expert thay vi "
          f"1, router + Linear ngoai khong doi) - "
          f"dang neu accuracy tang hoac on dinh "
          f"hon, khong dang neu chi tang nhe "
          f"tren du lieu toy de.")
    # C2: vector hoa vs vong lap, cung lop, cung
    # du lieu an (hidden cua tap test).
    torch.manual_seed(SEED)
    lop = MoELayer(D_MODEL, D_AN_MOE,
                   n_experts=N_EXPERTS, top_k=TOP_K)
    torch.manual_seed(SEED)
    mo_hinh_tam = MoEModel(top_k=TOP_K)
    with torch.no_grad():
        hidden = F.relu(mo_hinh_tam.fc1(x_test))
    sai = _sai_so_lon_nhat(lop(hidden)[0],
                           moe_forward_vectorized(
                               lop, hidden)[0])
    assert sai <= 1e-6, "C2: 2 ban phai bang nhau."
    t_vong = bench(lambda: lop(hidden))
    t_vec = bench(
        lambda: moe_forward_vectorized(lop, hidden))
    ti_le = t_vec / t_vong
    print(f"  C2 vector hoa (batch {len(x_test)}): "
          f"vong lap={t_vong:.2f} ms | vector hoa="
          f"{t_vec:.2f} ms | sai so={sai:.2e}")
    print(f"     Ket luan: "
          f"{_ket_luan_ti_le(ti_le, 'vector hoa', 'vong lap')}.")
    # C3: capacity 1.25 (theo de) va 0.5 (qua nho).
    ket_qua_cap = {}
    for he_so in (1.25, 0.5):
        torch.manual_seed(SEED)
        mo_hinh_cap = MoEModel(
            top_k=TOP_K, capacity_factor=he_so)
        kq_cap = train_model(mo_hinh_cap, x_train,
                             y_train, x_test, y_test)
        ket_qua_cap[he_so] = kq_cap["accuracy"]
        chenh_cap = kq_cap["accuracy"] - acc_moe_k1
        n_mau_cap = round(abs(chenh_cap)
                          * len(y_test))
        if abs(chenh_cap) < 0.005:
            cap = (f"accuracy gan nhu hoa "
                   f"({chenh_cap * 100:.1f} diem %, "
                   f"~{n_mau_cap} mau)")
        elif chenh_cap > 0:
            cap = (f"accuracy tang {chenh_cap * 100:.1f} "
                   f"diem % (~{n_mau_cap} mau)")
        else:
            cap = (f"accuracy giam "
                   f"{-chenh_cap * 100:.1f} diem % "
                   f"(~{n_mau_cap} mau)")
        print(f"  C3 capacity={he_so}: accuracy="
              f"{kq_cap['accuracy']:.3f} - {cap}.")
    print("     Capacity tinh theo batch hien tai: "
          "train 1600 token -> cap 500 (1.25) / "
          "200 (0.5) token/expert; test 400 token "
          "-> cap 125 / 50. Cap cang nho cang nhieu "
          "token bi roi (out=0) nen accuracy cang "
          "de giam.")
    return {"acc_k2": kq_k2["accuracy"],
            "pb_k2": pb_k2, "ti_le_vec": ti_le,
            "acc_cap": ket_qua_cap}


def tra_loi_tu_kiem_tra():
    """Tra loi ngan 3 cau tu kiem tra trong de."""
    print("\nTRA LOI TU KIEM TRA (3 cau trong de):")
    print("  1. top_k=1 van hoc duoc vi router hoc "
          "chuyen mon hoa: moi expert dan gioi 1 "
          "kieu token, 1 expert du cho 1 token. "
          "Danh doi lon nhat: on dinh kem - router "
          "chon sai la mat han, khong co expert "
          "thu 2 cuu, va de collapse hon top_k=2.")
    print("  2. aux = n * sum(f_i * P_i) nho nhat "
          "(=1) khi f_i = P_i = 1/n, lon (~n) khi "
          "don vao 1 expert; gradient day P_i ve "
          "deu. Nhan voi n de loss o trang thai "
          "can bang co thang ~1, khong bi task "
          "loss ngo lo.")
    print("  3. Suy luan re vi moi token chi kich "
          "hoat k expert (conditional compute). "
          "MoE dat hon khi: bo nho (phai chua "
          "toan bo N expert), giao tiep all-to-all "
          "khi expert o nhieu GPU, batch nho.")


def main():
    """Chay ca lab: tu kiem tra, 3 bai, tom tat."""
    print("Lab MoE Day 26 - "
          f"torch {torch.__version__}")
    print("Thiet bi: "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    tu_kiem_tra()
    torch.manual_seed(SEED)
    x_train, y_train, x_test, y_test = make_clusters()
    print(f"Du lieu: train {tuple(x_train.shape)}, "
          f"test {tuple(x_test.shape)} (4 cum, "
          f"moi cum 500 diem).")
    kq1 = bai1(x_train, y_train, x_test, y_test)
    kq2 = bai2(kq1["moe_model"], x_train, y_train,
               x_test, y_test)
    kq3 = bai3(x_train, y_train, x_test, y_test,
               kq1["moe_acc"])
    print("\nBANG KET QUA: Dense acc "
          f"{kq1['dense_acc']:.3f} (tham so "
          f"{kq1['ts_dense']}) | MoE top_k=1 acc "
          f"{kq1['moe_acc']:.3f} (tong "
          f"{kq1['ts_moe']}, kich hoat "
          f"{kq1['ts_kich_hoat']}) | phan bo truoc "
          f"aux max {max(kq2['pb_truoc']):.0f}% -> "
          f"sau aux max {max(kq2['pb_sau']):.0f}%, "
          f"acc sau aux {kq2['acc_sau']:.3f} | "
          f"top_k=2 acc {kq3['acc_k2']:.3f} | "
          f"vector hoa/vong lap "
          f"{kq3['ti_le_vec']:.2f}x | capacity 1.25 "
          f"acc {kq3['acc_cap'][1.25]:.3f}, 0.5 acc "
          f"{kq3['acc_cap'][0.5]:.3f}.")
    print("Ket luan chung: MoE chay duoc that "
          "(router -> top-k -> expert -> gop co "
          "trong so); collapse do duoc bang phan "
          "bo %/expert va aux loss day phan bo ve "
          "deu; tong tham so lon khong dong nghia "
          "FLOPs/token lon - chi k expert duoc "
          "kich hoat.")
    tra_loi_tu_kiem_tra()


if __name__ == "__main__":
    main()
