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
from torch.utils.data import DataLoader
from tqdm import tqdm

from nstsr.config.config import exp_dir, load_config
from nstsr.data.dataset import SARTripletDataset
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
        p.add_argument("--gpu", type=int, default=0)
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

        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"device = {device}")

        # ── dataset ───────────────────────────────────────────────────
        ratio_model = None
        if cfg.data.mode == "online_cs":
            ratio_model = build_ratio_denoiser(arch=args.ratio_arch, ckpt_path=args.ratio_ckpt, device=device)
        train_ds = SARTripletDataset(
            root=cfg.data.root, split="train", pol=cfg.pol,
            patch_size=cfg.data.patch_size, mode=cfg.data.mode,
            ratio_denoiser=ratio_model, augment=cfg.data.augment,
        )
        val_ds = SARTripletDataset(
            root=cfg.data.root, split="val", pol=cfg.pol,
            patch_size=cfg.data.patch_size, mode=cfg.data.mode,
            ratio_denoiser=ratio_model, augment=False,
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
        model = self._build_model(cfg, cfg.data.patch_size).to(device)
        schedule = make_linear_schedule(
            T=cfg.diffusion.T,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
            device=device,
        )
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        ema = EMA(model, decay=cfg.train.ema_decay)
        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"model params: {n_params/1e6:.2f} M")

        # ── resume ────────────────────────────────────────────────────
        start_iter = 0
        if args.resume is not None:
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["model"])
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
                ema.update(model)

            pbar.update(1)

            if (it + 1) % cfg.train.log_every == 0:
                avg = running_loss / cfg.train.log_every
                dt  = time.time() - t0
                rate = cfg.train.log_every / dt if dt > 0 else 0.0
                pbar.set_postfix(loss=f"{avg:.4f}", it_s=f"{rate:.1f}")
                tb.add_scalar("train/loss", avg, it + 1)
                running_loss = 0.0
                t0 = time.time()

            if (it + 1) % cfg.train.val_every == 0:
                self._validate(model, ema, schedule, val_loader, cfg, device, out_dir, it + 1, tb)

            if (it + 1) % cfg.train.ckpt_every == 0:
                self._save_ckpt(out_dir, model, ema, opt, it + 1, tag="last")
                tqdm.write(f"[{cfg.exp_name}] iter {it+1}  ckpt saved")

            it += 1

        pbar.close()
        self._save_ckpt(out_dir, model, ema, opt, it, tag="last")
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
        s  = batch["s"].to(device)
        cs = batch["cs"].to(device)
        x0_norm = dips_sample(
            eval_model, schedule, s, cs,
            shape=s.shape, device=device,
            S=cfg.sampling.steps, t_last=cfg.sampling.t_last,
            r=cfg.sampling.r, eta=cfg.sampling.eta,
        )
        # 시각화 (log-norm 도메인)
        save_grid_png(
            out_dir / "val" / f"step_{step:07d}.png",
            [batch["s"][0], batch["y"][0], batch["cs"][0], x0_norm[0].cpu()],
            titles=["s (clean)", "y (gt noisy)", "cs (ratio)", "sample"],
        )
        # linear domain mean / std 도 함께 로깅
        y_log_pred = denormalize_image(x0_norm.cpu().numpy(), pol=cfg.pol)
        y_lin_pred = np.power(10.0, y_log_pred)
        tb.add_scalar("val/pred_mean_linear", float(y_lin_pred.mean()), step)
        tb.add_scalar("val/pred_std_linear",  float(y_lin_pred.std()), step)
        self.logger.info(f"  val @ step {step}: pred mean={y_lin_pred.mean():.4g} std={y_lin_pred.std():.4g}")

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
