import numpy as np, os, glob, time, csv, warnings
warnings.filterwarnings('ignore')
K="/media/sdb8TB/sentinel1/korea"
OUT=f"{K}/speckle_ds"; os.makedirs(OUT,exist_ok=True); os.makedirs(f"{OUT}/r_patches",exist_ok=True); os.makedirs(f"{OUT}/cmask_patches",exist_ok=True); os.makedirs(f"{OUT}/idx_patches",exist_ok=True); os.makedirs(f"{OUT}/splits",exist_ok=True)
NR,NC=13124,68647; NCt=(NC//16)*16; eps=1e-12
P=512; STR=512; BAD=(4100,4700); BLK=1024
mu_f=f"{K}/ave_dir/C11.img"; msr_f=f"{K}/ps_work/MSR.img"; val_f=f"{K}/mask/valid_mask.img"
VVDIR=f"{K}/rslc_prep_vv"

# --- 30장 균등 ---
allr=sorted(glob.glob(f"{VVDIR}/*.rslc"))
sel=sorted(set(np.round(np.linspace(0,len(allr)-1,30)).astype(int).tolist()))
dates=[os.path.basename(allr[i])[:8] for i in sel]
print(f"{len(dates)} dates: {dates[0]}..{dates[-1]}",flush=True)

# --- 공유: μ, D_A, valid (full, NCt) ---
mu=np.memmap(mu_f,'>f4','r',shape=(NR,NC))[:,:NCt]
msr=np.memmap(msr_f,'>f4','r',shape=(NR,NC))[:,:NCt]
val=np.memmap(val_f,'>f4','r',shape=(NR,NC))[:,:NCt]

# 공유 valid = ls geometry valid AND μ>0(데이터 footprint), BAD밴드 제외.
mu_full=np.asarray(mu).astype(np.float32)            # r 계산·footprint 용
vv=((np.asarray(val)>0.5)&(mu_full>0)).astype(np.float32); vv[BAD[0]:BAD[1],:]=0.0

# --- patch 좌표: vv(=ls&μ>0) >0.7, BAD밴드 미교차 ---
coords=[]
for y0 in range(0,NR-P+1,STR):
    if y0<BAD[1] and y0+P>BAD[0]: continue
    for x0 in range(0,NCt-P+1,STR):
        if vv[y0:y0+P,x0:x0+P].mean()>0.7:
            coords.append((y0,x0))
N=len(coords); print(f"valid patches: {N}",flush=True)
with open(f"{OUT}/coords.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["idx","y0","x0"]); [w.writerow([k,y,x]) for k,(y,x) in enumerate(coords)]
# 공유 cond 저장
def stk(a): return np.stack([np.asarray(a[y:y+P,x:x+P]) for (y,x) in coords]).astype(np.float32)
np.save(f"{OUT}/mu_patches.npy", stk(mu))
da=1.0/np.maximum(np.asarray(msr),1e-3)
np.save(f"{OUT}/da_patches.npy", np.stack([da[y:y+P,x:x+P] for (y,x) in coords]).astype(np.float32))
np.save(f"{OUT}/valid_patches.npy", np.stack([vv[y:y+P,x:x+P] for (y,x) in coords]).astype(np.float32))
print("saved shared mu/da/valid patches",flush=True)

P4=P//4; P16=P//16
def ml(a,b,W):
    R=a.shape[0]//b
    return np.where.__self__ if False else None
def mlk(x,m,b,Wb):
    R=x.shape[0]//b
    x=x[:R*b,:Wb*b].reshape(R,b,Wb,b); m=m[:R*b,:Wb*b].reshape(R,b,Wb,b)
    num=(x*m).sum((1,3)); den=m.sum((1,3))
    return np.where(den>0,num/np.maximum(den,1),np.nan).astype(np.float32)
W4,W16=NCt//4,NCt//16
samples=[]
for di,d in enumerate(dates):
    t0=time.time()
    # full-res r
    r=np.zeros((NR,NCt),np.float32); vmask=np.zeros((NR,NCt),np.float32)
    fr=open(f"{VVDIR}/{d}.rslc",'rb')
    for s in range(0,NR,BLK):
        nb=min(BLK,NR-s); cnt=nb*NC
        rs=np.fromfile(fr,dtype='>c8',count=cnt).reshape(nb,NC)[:,:NCt]
        y=rs.real.astype(np.float32)**2+rs.imag.astype(np.float32)**2
        mm=mu_full[s:s+nb]; vm=((y>0)&(mm>0)).astype(np.float32)
        r[s:s+nb]=np.where(vm>0,np.log10((y+eps)/(mm+eps)),0.0); vmask[s:s+nb]=vm
    fr.close()
    r4=mlk(r,vmask,4,W4); r16=mlk(r,vmask,16,W16)
    med=np.nanmedian(r4); mad=np.nanmedian(np.abs(r4-med))*1.4826
    # per-date ratio 보정
    def pstat(y0,x0):
        a4=r4[y0//4:y0//4+P4, x0//4:x0//4+P4]; a16=r16[y0//16:y0//16+P16, x0//16:x0//16+P16]
        return np.nanmean(a4),np.nanstd(a4),np.nanstd(a16),a4
    ratios=[]; means=[]
    for (y0,x0) in coords:
        m,s4,s16,_=pstat(y0,x0); ratios.append(s4/max(s16,1e-6)); means.append(m)
    ratios=np.array(ratios); means=np.array(means)
    rmed=np.nanmedian(ratios); tau_ratio=rmed*0.6; tau_mean=4*mad; tau_cell=4*mad
    acc_idx=[]; rst=[]; cst=[]
    for k,(y0,x0) in enumerate(coords):
        m,s4,s16,a4=pstat(y0,x0); ratio=s4/max(s16,1e-6)
        if np.isfinite(a4).mean()<0.5: continue                   # 이 날짜 데이터 부족(no-data) → skip
        if (abs(m-med)>tau_mean) or (ratio<tau_ratio): continue   # patch 기각(변화)
        # per-pixel change-mask: a4 셀 이탈 → 제외, ×4 업샘플
        cellbad=(np.abs(a4-med)>tau_cell)
        cellbad=np.repeat(np.repeat(cellbad,4,0),4,1)[:P,:P]
        vp=vv[y0:y0+P,x0:x0+P]                                     # 공유 valid(ls&μ>0)
        vmp=vmask[y0:y0+P,x0:x0+P]                                 # 이 날짜 데이터(y>0&μ>0)
        cm=((vp>0.5)&(vmp>0.5)&(~np.nan_to_num(cellbad,nan=True).astype(bool))).astype(np.uint8)
        if cm.mean()<0.3: continue                                # 거의 다 마스크면 skip
        acc_idx.append(k); rst.append(r[y0:y0+P,x0:x0+P].copy()); cst.append(cm)
        samples.append(f"{d}:{len(acc_idx)-1}")
    np.save(f"{OUT}/r_patches/{d}.npy", np.stack(rst).astype(np.float32))
    np.save(f"{OUT}/cmask_patches/{d}.npy", np.stack(cst).astype(np.uint8))
    np.save(f"{OUT}/idx_patches/{d}.npy", np.array(acc_idx,np.int32))
    print(f" [{di+1}/{len(dates)}] {d}: accept {len(acc_idx)}/{N}  ({time.time()-t0:.0f}s)  med={med:+.3f} tau_r={tau_ratio:.2f}",flush=True)

# splits
rng=np.random.RandomState(42); rng.shuffle(samples)
nval=max(1,len(samples)//10)
open(f"{OUT}/splits/val.txt","w").write("\n".join(samples[:nval]))
open(f"{OUT}/splits/train.txt","w").write("\n".join(samples[nval:]))
print(f"DONE. samples={len(samples)}  train={len(samples)-nval} val={nval}",flush=True)
print(f"out: {OUT}",flush=True)
