"""
TrainPipeline — diffusion 모델 학습.

CLI:
    python main.py train --config configs/vv_default.yaml
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from nstsr.config.config import exp_dir, load_config
from nstsr.data.dataset import SARSpeckleDataset
from nstsr.diffusion.schedule import make_linear_schedule
from nstsr.diffusion.trainer import training_step
from nstsr.diffusion.sampler import sample as dips_sample
from nstsr.model.ratio_denoiser import build_ratio_denoiser
from nstsr.model.unet import UNet
from nstsr.utils.logger import TBWriter, get_logger
from nstsr.utils.viz import save_grid_png
from nstsr.config.norm_config import denormalize_image


class EMA:
    """exponential moving average — store ema weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class TrainPipeline:
    def __init__(self):
        self.logger = get_logger("nstsr.train")

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="train diffusion model")
        p.add_argument("--config", default=None, help="YAML config 경로 (생략 시 configs/vv_default.yaml)")
        p.add_argument("--gpu", type=int, default=0, help="단일 GPU index (기본)")
        p.add_argument("--gpus", type=int, nargs="+", default=None,
                       help="여러 GPU 지정 시 nn.DataParallel (예: --gpus 0 1, 최대 2장). "
                            "생략 시 --gpu 한 장만 사용.")
        p.add_argument("--ratio_arch", default="identity")
        p.add_argument("--ratio_ckpt", default=None,
                       help="online_cs mode 일 때 필요. cache_cs 면 무시.")
        p.add_argument("--resume", default=None, help="last.pt 등 체크포인트 경로 (이어서 학습)")
        p.add_argument("--data_root", default=None,
                       help="cfg.data.root 오버라이드 (다른 머신에서 data_root 경로를 yaml 수정 없이 지정)")
        return p

    # ─────────────────────────────────────────────────────────────────
    def _build_model(self, cfg, input_resolution: int) -> UNet:
        m = cfg.model
        return UNet(
            in_ch=m.in_ch,
            out_ch=m.out_ch,
            base_ch=m.base_ch,
            ch_mults=tuple(m.ch_mults),
            num_res_blocks=m.num_res_blocks,
            attn_resolutions=tuple(m.attn_resolutions),
            use_mcam=m.use_mcam,
            input_resolution=input_resolution,
        )

    # ─────────────────────────────────────────────────────────────────
    def run(self, args: argparse.Namespace | None = None) -> None:
        if args is None:
            args = self.build_parser().parse_args()

        cfg = load_config(args.config)
        if args.data_root is not None:
            cfg.data.root = args.data_root
            self.logger.info(f"data.root overridden → {cfg.data.root}")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # ── device / GPU 선택 (기본 단일, --gpus 지정 시 DataParallel) ──
        use_cuda = torch.cuda.is_available()
        if use_cuda and args.gpus:
            n_dev = torch.cuda.device_count()
            gpu_ids = [g for g in args.gpus if 0 <= g < n_dev]
            if len(gpu_ids) != len(args.gpus):
                self.logger.warning(f"존재하지 않는 GPU id 무시 (가용 {n_dev}장): "
                                    f"{sorted(set(args.gpus) - set(gpu_ids))}")
            if len(gpu_ids) > 2:
                self.logger.warning(f"DataParallel 최대 2장 권장 — 앞 2개만 사용: {gpu_ids[:2]}")
                gpu_ids = gpu_ids[:2]
            if not gpu_ids:
                gpu_ids = [args.gpu]
        else:
            gpu_ids = [args.gpu] if use_cuda else []
        device = f"cuda:{gpu_ids[0]}" if use_cuda else "cpu"
        multi_gpu = len(gpu_ids) > 1
        self.logger.info(f"device = {device}"
                         + (f"  +DataParallel{gpu_ids}" if multi_gpu else ""))

        # ── dataset (renoise speckle) ─────────────────────────────────
        train_ds = SARSpeckleDataset(
            root=cfg.data.root, split="train",
            patch_size=cfg.data.patch_size, augment=cfg.data.augment,
        )
        val_ds = SARSpeckleDataset(
            root=cfg.data.root, split="val",
            patch_size=cfg.data.patch_size, augment=False,
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.data.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0,
        )
        self.logger.info(f"dataset: train={len(train_ds)} val={len(val_ds)}")

        # ── model + schedule + opt ────────────────────────────────────
        # core_model: 원본 (EMA / state_dict / 저장·로드 용 — module. prefix 없음)
        # model:      forward 용 (multi-GPU 면 DataParallel 래핑)
        core_model = self._build_model(cfg, cfg.data.patch_size).to(device)
        model = torch.nn.DataParallel(core_model, device_ids=gpu_ids) if multi_gpu else core_model
        schedule = make_linear_schedule(
            T=cfg.diffusion.T,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            device=device,
        )
        opt = torch.optim.Adam(core_model.parameters(), lr=cfg.train.lr)
        ema = EMA(core_model, decay=cfg.train.ema_decay)
        n_params = sum(p.numel() for p in core_model.parameters())
        self.logger.info(f"model params: {n_params/1e6:.2f} M")

        # ── resume ────────────────────────────────────────────────────
        start_iter = 0
        if args.resume is not None:
            ck = torch.load(args.resume, map_location=device)
            core_model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
            start_iter = ck.get("iter", 0)
            self.logger.info(f"resumed from {args.resume} @ iter {start_iter}")

        out_dir = exp_dir(cfg)
        tb = TBWriter(out_dir / "tb")

        # ── training loop ─────────────────────────────────────────────
        model.train()
        it = start_iter
        accum = max(1, cfg.train.grad_accum_steps)
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0
        t0 = time.time()
        loader_iter = iter(train_loader)
        train_hist: list[tuple[int, float]] = []   # (step, train_loss)
        val_hist:   list[tuple[int, float]] = []    # (step, val_loss)
        pbar = tqdm(total=cfg.train.total_iters, initial=start_iter,
                    desc=cfg.exp_name, dynamic_ncols=True, unit="it")

        while it < cfg.train.total_iters:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            loss = training_step(batch, model, schedule, device=device) / accum
            loss.backward()
            running_loss += loss.item() * accum

            if ((it + 1) % accum) == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                ema.update(core_model)

            pbar.update(1)

            if (it + 1) % cfg.train.log_every == 0:
                avg = running_loss / cfg.train.log_every
                dt  = time.time() - t0
                rate = cfg.train.log_every / dt if dt > 0 else 0.0
                pbar.set_postfix(loss=f"{avg:.4f}", it_s=f"{rate:.1f}")
                tb.add_scalar("train/loss", avg, it + 1)
                train_hist.append((it + 1, avg))
                running_loss = 0.0
                t0 = time.time()

            if (it + 1) % cfg.train.val_every == 0:
                vloss = self._val_loss(model, schedule, val_loader, device)
                tb.add_scalar("val/loss", vloss, it + 1)
                val_hist.append((it + 1, vloss))
                self._validate(core_model, ema, schedule, val_loader, cfg, device, out_dir, it + 1, tb)
                self._plot_loss_curve(out_dir, train_hist, val_hist)
                tqdm.write(f"[{cfg.exp_name}] iter {it+1}  val_loss={vloss:.4f}  "
                           f"→ loss_curve.png 갱신")

            if (it + 1) % cfg.train.ckpt_every == 0:
                self._save_ckpt(out_dir, core_model, ema, opt, it + 1, tag="last")
                tqdm.write(f"[{cfg.exp_name}] iter {it+1}  ckpt saved")

            it += 1

        pbar.close()
        self._save_ckpt(out_dir, core_model, ema, opt, it, tag="last")
        tb.close()
        self.logger.info(f"training done. out_dir={out_dir}")

    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self, model, ema, schedule, val_loader, cfg, device, out_dir, step, tb):
        if len(val_loader) == 0:
            return
        # EMA weights 로 inference
        eval_model = self._build_model(cfg, cfg.data.patch_size).to(device)
        ema.copy_to(eval_model)
        eval_model.eval()

        batch = next(iter(val_loader))
        cond = batch["cond"].to(device)
        r_hat = dips_sample(
            eval_model, schedule, cond,
            shape=(cond.shape[0], 1, cond.shape[2], cond.shape[3]), device=device,
            S=cfg.sampling.steps, t_last=cfg.sampling.t_last,
            r=cfg.sampling.r, eta=cfg.sampling.eta,
        )

        def _stretch(t, lo=1.0, hi=99.0):
            a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            vmn, vmx = np.percentile(a, [lo, hi])
            return float(vmn), float(vmx + (1e-6 if vmx <= vmn else 0.0))

        mu_n = cond[0, 0:1].cpu()        # 구조(μ_norm)
        r_gt = batch["r"][0].cpu()       # target speckle r
        r_s  = r_hat[0].cpu()            # 생성 speckle r̂
        v_mu = _stretch(mu_n); v_r = (-1.0, 1.0)   # r 은 [-1,1] 공통 스케일
        save_grid_png(
            out_dir / "val" / f"step_{step:07d}.png",
            [mu_n, r_gt, r_s],
            titles=["μ (structure)", "r gt (speckle)", "r̂ (sampled speckle)"],
            vranges=[v_mu, v_r, v_r],
        )
        # speckle 통계 비교 (생성 r̂ std 가 gt r std 를 따라가야 함)
        tb.add_scalar("val/r_gt_std",  float(r_gt.std()),  step)
        tb.add_scalar("val/r_hat_std", float(r_s.std()),   step)
        self.logger.info(f"  val @ step {step}: r_gt std={r_gt.std():.4f}  r_hat std={r_s.std():.4f}")

    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_loss(self, model, schedule, val_loader, device, max_batches: int = 64) -> float:
        """Held-out ε-MSE (training loss 와 동일 정의). 고정 seed 로 (t, ε) 를 뽑아
        eval 간 비교 가능한 안정적 곡선을 만든다. EMA 가 아닌 학습 weight 로 평가."""
        if len(val_loader) == 0:
            return float("nan")
        was_training = model.training
        model.eval()
        g = torch.Generator().manual_seed(1234)   # CPU generator, 고정
        total, n = 0.0, 0
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            r     = batch["r"].to(device)
            cond  = batch["cond"].to(device)
            cmask = batch["cmask"].to(device)
            B = r.size(0)
            t   = torch.randint(0, schedule.T, (B,), generator=g).to(device)
            eps = torch.randn(r.shape, generator=g).to(device)
            sqrt_ab   = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
            sqrt_1_ab = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
            x_t = sqrt_ab * r + sqrt_1_ab * eps
            se = (model(x_t, t, cond) - eps) ** 2 * cmask
            total += (se.sum() / cmask.sum().clamp_min(1.0)).item()
            n += 1
        if was_training:
            model.train()
        return total / max(n, 1)

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _plot_loss_curve(out_dir: Path, train_hist, val_hist) -> None:
        """매 eval 마다 train/val loss 곡선을 loss_curve.png 로 덮어쓴다."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        if train_hist:
            ts, ls = zip(*train_hist)
            ax.plot(ts, ls, color="tab:blue", alpha=0.5, lw=1, label="train")
        if val_hist:
            vs, vl = zip(*val_hist)
            ax.plot(vs, vl, color="tab:red", marker="o", ms=3, label="val")
        ax.set_xlabel("iter")
        ax.set_ylabel("ε-MSE loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=110)
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────
    def _save_ckpt(self, out_dir: Path, model, ema, opt, it: int, tag: str = "last") -> None:
        ck = {
            "iter": it,
            "model": model.state_dict(),
            "ema":   ema.state_dict(),
            "opt":   opt.state_dict(),
        }
        torch.save(ck, out_dir / f"{tag}.pt")
        # EMA-only (추론용)
        torch.save({"model": ema.state_dict(), "iter": it}, out_dir / "ema.pt")
        self.logger.info(f"  ckpt saved: {tag}.pt + ema.pt @ iter {it}")
