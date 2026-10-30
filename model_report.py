"""
model_report.py

Comprehensive multi-algorithm benchmark on the REAL Digital Heart Twin / MSCardio data.
Benchmarks THREE clearly-labelled evaluation paths so no result can be misattributed:

  PATH A — Classical features (48 hand-crafted SCG/HRV stats)
            + Mahalanobis / OCSVM / IsoForest / LOF / kNN anomaly scorers
            + Cosine / Euclidean / Mahalanobis / kNN / LogReg / RF verification

  PATH B — Deep-learning embeddings (256-d pooled output of the MAE foundation encoder)
            + the same set of anomaly/verification scorers as Path A
            (This shows encoder representation quality WITHOUT the VAE head.)

  PATH C — VAE/KL personal twin  ← THE CORE INNOVATION
            PersonalHeartTwin.anomaly_score() called directly on held-out windows.
            Genuine scores = this user's own held-out recordings.
            Impostor scores = recordings from every other trained subject.
            Requires at least one trained twin_user_*.pt checkpoint.

Writes checkpoints/model_report.json.
Run:  python model_report.py --data_path Digital Heart Twin
"""
from __future__ import annotations

import argparse, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif

from src.config import CHECKPOINT_DIR, ModelConfig, get_device
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset, _load_scg_csv
from src.evaluation.classical_baseline import extract_features
from src.utils.preprocessing import resample_signal


# ---------------------------------------------------------------------------
def feature_names():
    base = ['std', 'rms', 'iqr', 'skew', 'kurt', 'zcr', 'bp0.8-4', 'bp4-12', 'bp12-30', 'centroid', 'hrpeak']
    names = []
    for ax in ['x', 'y', 'z', 'mag']:
        names += [f'{ax}_{b}' for b in base]
    names += ['hr', 'sdnn', 'rmssd', 'ac_strength']
    return names


def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def eer_auc(genuine, impostor):
    genuine, impostor = np.asarray(genuine, float), np.asarray(impostor, float)
    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    s = -np.concatenate([genuine, impostor])           # higher = genuine-like
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return float('nan'), float('nan')
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2), float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Verification algorithms (LORO). Each returns genuine/impostor score arrays.
# ---------------------------------------------------------------------------
def _templates(X, subj, uniq):
    return {s: X[subj == s].mean(0) for s in uniq}


def verify_template(X, subj, metric='cosine'):
    subj = np.asarray(subj); uniq = np.unique(subj)
    if metric == 'cosine':
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        def d(a, t): return 1 - float(a @ t / (np.linalg.norm(t) + 1e-9))
        Xu = Xn
    elif metric == 'euclidean':
        def d(a, t): return float(np.linalg.norm(a - t))
        Xu = X
    else:  # mahalanobis (global)
        cov = np.cov(X.T) + 1e-3 * np.eye(X.shape[1]); P = np.linalg.pinv(cov)
        def d(a, t): z = a - t; return float(np.sqrt(z @ P @ z))
        Xu = X
    tmpl = _templates(Xu, subj, uniq)
    g, imp = [], []
    for s in uniq:
        idx = np.where(subj == s)[0]
        if len(idx) < 2: continue
        for i in idx:
            own = Xu[idx[idx != i]].mean(0)
            g.append(d(Xu[i], own))
            for s2 in uniq:
                if s2 != s: imp.append(d(Xu[i], tmpl[s2]))
    return np.array(g), np.array(imp)


def verify_knn(X, subj):
    subj = np.asarray(subj); uniq = np.unique(subj)
    g, imp = [], []
    for s in uniq:
        idx = np.where(subj == s)[0]
        if len(idx) < 2: continue
        for i in idx:
            same = idx[idx != i]
            g.append(float(np.min(np.linalg.norm(X[same] - X[i], axis=1))))
            for s2 in uniq:
                if s2 == s: continue
                oth = np.where(subj == s2)[0]
                imp.append(float(np.min(np.linalg.norm(X[oth] - X[i], axis=1))))
    return np.array(g), np.array(imp)


def verify_supervised(X, subj, clf_factory):
    """Learned pairwise verifier on |probe - template| difference vectors,
    subject-disjoint 5-fold (train templates/classifier on train subjects,
    test on held-out subjects). Score = 1 - P(same)."""
    subj = np.asarray(subj); uniq = np.unique(subj)
    rng = np.random.RandomState(0)
    folds = np.array_split(rng.permutation(uniq), 5)
    g, imp = [], []
    for fk in range(5):
        test_s = set(folds[fk].tolist()); train_s = [s for s in uniq if s not in test_s]
        # build training diff-vectors from train subjects
        tt = {s: X[subj == s].mean(0) for s in train_s}
        Dtr, ytr = [], []
        for s in train_s:
            idx = np.where(subj == s)[0]
            if len(idx) < 2: continue
            for i in idx:
                own = X[idx[idx != i]].mean(0)
                Dtr.append(np.abs(X[i] - own)); ytr.append(1)
                for s2 in rng.choice(train_s, size=min(3, len(train_s)), replace=False):
                    if s2 != s: Dtr.append(np.abs(X[i] - tt[s2])); ytr.append(0)
        if len(set(ytr)) < 2: continue
        clf = clf_factory().fit(np.array(Dtr), np.array(ytr))
        tt_te = {s: X[subj == s].mean(0) for s in test_s}
        for s in test_s:
            idx = np.where(subj == s)[0]
            if len(idx) < 2: continue
            for i in idx:
                own = X[idx[idx != i]].mean(0)
                g.append(1 - float(clf.predict_proba([np.abs(X[i] - own)])[0, 1]))
                for s2 in test_s:
                    if s2 != s: imp.append(1 - float(clf.predict_proba([np.abs(X[i] - tt_te[s2])])[0, 1]))
    return np.array(g), np.array(imp)


# ---------------------------------------------------------------------------
# Anomaly algorithms (per user: baseline = first N recs; held-out own vs others)
# ---------------------------------------------------------------------------
def anomaly(X, subj, model='mahalanobis', n_base=5, min_rec=6):
    subj = np.asarray(subj); uniq = np.unique(subj)
    g, imp, per = [], [], {}
    others_all = X
    for s in uniq:
        idx = np.where(subj == s)[0]
        if len(idx) < min_rec: continue
        base = X[idx[:n_base]]; held = idx[n_base:]
        oth = np.where(subj != s)[0]
        if model == 'mahalanobis':
            mu = base.mean(0); sd = X.std(0) + 1e-6
            sc = lambda z: float(np.sqrt(np.mean(((z - mu) / sd) ** 2)))
        elif model == 'ocsvm':
            clf = OneClassSVM(nu=0.3, gamma='scale').fit(base); sc = lambda z: -float(clf.decision_function([z])[0])
        elif model == 'isoforest':
            clf = IsolationForest(n_estimators=100, random_state=0).fit(base); sc = lambda z: -float(clf.decision_function([z])[0])
        elif model == 'lof':
            k = max(1, min(len(base) - 1, 3)); clf = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(base); sc = lambda z: -float(clf.decision_function([z])[0])
        else:  # knn to baseline
            sc = lambda z: float(np.min(np.linalg.norm(base - z, axis=1)))
        gs = [sc(X[i]) for i in held]; ims = [sc(X[i]) for i in oth]
        g += gs; imp += ims
        e, a = eer_auc(gs, ims); per[str(s)] = {'eer': e, 'auc': a, 'n_held': len(gs)}
    pe, pa = eer_auc(g, imp)
    me = float(np.nanmean([v['eer'] for v in per.values()])) if per else float('nan')
    return {'pooled_eer': pe, 'pooled_auc': pa, 'mean_per_user_eer': me, 'n_eligible': len(per), 'per_user': per}


# ---------------------------------------------------------------------------
def run_task_suite(X, subj, tag):
    print(f"[{tag}] verification algorithms…")
    ver = {}
    for name, fn in [('cosine-template', lambda: verify_template(X, subj, 'cosine')),
                     ('euclid-template', lambda: verify_template(X, subj, 'euclidean')),
                     ('mahalanobis-template', lambda: verify_template(X, subj, 'mahalanobis')),
                     ('knn', lambda: verify_knn(X, subj)),
                     ('logreg-pair', lambda: verify_supervised(X, subj, lambda: LogisticRegression(max_iter=300))),
                     ('rf-pair', lambda: verify_supervised(X, subj, lambda: RandomForestClassifier(n_estimators=120, random_state=0)))]:
        gg, ii = fn(); e, a = eer_auc(gg, ii); ver[name] = {'eer': e, 'auc': a}
        print(f"    {name:22} EER={e:.4f} AUC={a:.4f}")
    print(f"[{tag}] anomaly algorithms…")
    ano = {}
    for name in ['mahalanobis', 'ocsvm', 'isoforest', 'lof', 'knn']:
        r = anomaly(X, subj, name); ano[name] = {k: r[k] for k in ('pooled_eer', 'pooled_auc', 'mean_per_user_eer', 'n_eligible')}
        print(f"    {name:22} pooledEER={r['pooled_eer']:.4f} meanUserEER={r['mean_per_user_eer']:.4f}")
    return {'verification': ver, 'anomaly': ano}


def dl_embeddings(m, device):
    import torch
    from src.models.encoder import SCGEncoder
    enc = SCGEncoder(ModelConfig())
    ckpt = CHECKPOINT_DIR / "encoder_real.pt"
    if not ckpt.exists():
        print("[dl] encoder_real.pt missing — pretraining briefly…")
        from src.training.pretrain import run_pretraining
        run_pretraining(data_path='Digital Heart Twin', epochs=8, batch_size=64, checkpoint_name="encoder_real.pt", max_steps=250)
    enc.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)); enc.to(device).eval()
    X, subj = [], []
    rows = list(m.iterrows())
    for i, (_, row) in enumerate(rows):
        df = m[(m["subject_id"] == row["subject_id"]) & (m["recording_id"] == row["recording_id"])]
        ds = SCGWindowDataset(df)
        if len(ds) == 0: continue
        xs = torch.stack([ds[j][0] for j in range(len(ds))], 0).to(device)
        with torch.no_grad():
            e = enc.encode_pooled(xs).mean(0).cpu().numpy()
        X.append(e); subj.append(str(row["subject_id"]))
        if (i + 1) % 150 == 0: print(f"  [dl-emb] {i+1}/{len(rows)}")
    return standardize(np.vstack(X)), subj


def vae_kl_benchmark(manifest_df, checkpoint_dir=CHECKPOINT_DIR):
    """
    PATH C — Full Personal Heart Twin (VAE/KL Divergence Anomaly Scorer).
    Evaluates PersonalHeartTwin.anomaly_score() on held-out recordings vs impostor recordings.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.utils.anomaly_scorer import load_model_bundle, load_user_twin
    from src.data_loader import create_session_split
    from src.dataset import collate_scg

    device = get_device()
    twin_files = sorted(Path(checkpoint_dir).glob("twin_user_*.pt"))
    if not twin_files:
        print("[vae-kl] No twin_user_*.pt checkpoints found.")
        return None

    try:
        bundle = load_model_bundle(checkpoint_dir, device)
        foundation = bundle["foundation"]
    except Exception as ex:
        print(f"[vae-kl] Could not load foundation bundle: {ex}")
        return None

    twin_subjects = [p.stem.replace("twin_user_", "") for p in twin_files]
    manifest_sids = set(manifest_df["subject_id"].astype(str).unique())
    eligible_sids = [s for s in twin_subjects if s in manifest_sids]

    heldout_windows: dict[str, torch.Tensor] = {}
    for sid in eligible_sids:
        sub_df = manifest_df[manifest_df["subject_id"].astype(str) == sid].copy()
        if len(sub_df) < 2:
            continue
        enroll_df, test_df = create_session_split(sub_df, min_enrollment=3)
        if test_df.empty:
            test_df = sub_df.iloc[len(sub_df) // 2 :]
        if test_df.empty:
            continue
        ds = SCGWindowDataset(test_df, min_quality=0.35)
        if len(ds) == 0:
            ds = SCGWindowDataset(test_df, min_quality=0.0)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=min(40, len(ds)), shuffle=False, collate_fn=collate_scg)
        x, _ = next(iter(loader))
        heldout_windows[sid] = x.to(device)

    per_user = {}
    pooled_gen, pooled_imp = [], []
    for sid, x_gen in heldout_windows.items():
        twin = load_user_twin(sid, foundation, checkpoint_dir, device)
        if twin is None or not bool(twin.has_baseline.item()):
            continue
        with torch.no_grad():
            gen_scores = twin.anomaly_score(x_gen).cpu().numpy().tolist()
        imp_scores_list = []
        for other_sid, other_x in heldout_windows.items():
            if other_sid == sid:
                continue
            with torch.no_grad():
                s_imp = twin.anomaly_score(other_x).cpu().numpy().tolist()
            imp_scores_list.extend(s_imp)
        if not imp_scores_list or not gen_scores:
            continue
        pooled_gen.extend(gen_scores)
        pooled_imp.extend(imp_scores_list)
        e, a = eer_auc(gen_scores, imp_scores_list)
        per_user[sid] = {"eer": e, "auc": a, "n_gen": len(gen_scores), "n_imp": len(imp_scores_list)}

    if not per_user:
        return None

    pe, pa = eer_auc(pooled_gen, pooled_imp)
    valid_eers = [v["eer"] for v in per_user.values() if np.isfinite(v["eer"])]
    valid_aucs = [v["auc"] for v in per_user.values() if np.isfinite(v["auc"])]
    me = float(np.mean(valid_eers)) if valid_eers else float("nan")
    ma = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    return {
        "pooled_eer": pe,
        "pooled_auc": pa,
        "mean_per_user_eer": me,
        "mean_per_user_auc": ma,
        "n_evaluated_twins": len(per_user),
        "per_user": per_user,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="Digital Heart Twin")
    ap.add_argument("--out_path", default="checkpoints/model_report.json")
    args = ap.parse_args()
    device = get_device(); print(f"[report] device={device}")
    m = build_multimodal_manifest(raw_data_path=args.data_path)
    print(f"[report] {len(m)} recordings, {m['subject_id'].nunique()} subjects")

    # ---- PATH A: classical features ----
    print("[report] extracting Path A: classical features…")
    Xc, sc, durations = [], [], []
    for i, (_, row) in enumerate(m.iterrows()):
        v = extract_features(row["cal_scg"])
        if v is not None:
            Xc.append(v); sc.append(str(row["subject_id"]))
            try:
                a, fs = _load_scg_csv(row["cal_scg"]); durations.append(a.shape[-1] / fs)
            except Exception: pass
        if (i + 1) % 150 == 0: print(f"  [feat] {i+1}/{len(m)}")
    Xc = standardize(np.vstack(Xc))
    print(f"[report] classical matrix {Xc.shape}")

    res_c = run_task_suite(Xc, sc, "classical")

    # ---- feature importance (identity discrimination) ----
    print("[report] feature importance…")
    names = feature_names()
    y = np.array(sc)
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xc, y)
    mi = mutual_info_classif(Xc, y, random_state=0)
    imp = sorted(zip(names, rf.feature_importances_.tolist(), mi.tolist()), key=lambda t: -t[1])[:12]
    feat_imp = [{'feature': n, 'rf_importance': round(r, 4), 'mutual_info': round(float(mm), 3)} for n, r, mm in imp]
    for f in feat_imp[:8]: print(f"    {f['feature']:14} rf={f['rf_importance']:.4f} mi={f['mutual_info']:.3f}")

    # ---- PATH B: DL embeddings ----
    print("[report] extracting Path B: DL embeddings…")
    Xd, sd = dl_embeddings(m, device)
    print(f"[report] DL matrix {Xd.shape}")
    res_d = run_task_suite(Xd, sd, "deep-learning")

    # ---- PATH C: VAE/KL Personal Twin ----
    print("[report] evaluating Path C: PersonalHeartTwin VAE/KL anomaly scorer…")
    res_twin = vae_kl_benchmark(m)
    if res_twin:
        print(f"    VAE/KL personal twin: pooledEER={res_twin['pooled_eer']:.4f} meanUserEER={res_twin['mean_per_user_eer']:.4f} (twins={res_twin['n_evaluated_twins']})")

    # ---- best per task, per feature set ----
    def best_ver(d): return min(d['verification'].items(), key=lambda kv: kv[1]['eer'])
    def best_ano(d): return min(d['anomaly'].items(), key=lambda kv: kv[1]['pooled_eer'])
    bc_v, bd_v = best_ver(res_c), best_ver(res_d)
    bc_a, bd_a = best_ano(res_c), best_ano(res_d)

    report = {
        "dataset": {"n_recordings": int(Xc.shape[0]), "n_subjects": int(len(set(sc))),
                    "classical_dim": int(Xc.shape[1]), "dl_dim": int(Xd.shape[1]),
                    "duration_s": {"median": float(np.median(durations)), "min": float(np.min(durations)), "max": float(np.max(durations))} if durations else {}},
        "path_a_classical": res_c,
        "path_b_deep_learning_embeddings": res_d,
        "path_c_personal_twin_vae_kl": res_twin,
        "feature_importance_top": feat_imp,
        "best": {
            "classical_verify": [bc_v[0], bc_v[1]], "dl_verify": [bd_v[0], bd_v[1]],
            "classical_anomaly": [bc_a[0], bc_a[1]], "dl_anomaly": [bd_a[0], bd_a[1]],
            "personal_twin_anomaly": res_twin,
        },
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out_path, "w"), indent=2)

    print("\n================  MULTI-ALGORITHM BENCHMARK (real Digital Heart Twin)  ================")
    print(f"{'Method / Path':<32}{'Best / Core Scorer':<24}{'EER':>8}{'AUC':>8}")
    print(f"{'Path A: Classical (48 stats)':<32}{bc_a[0]:<24}{bc_a[1]['pooled_eer']:>8.4f}{bc_a[1]['pooled_auc']:>8.4f}")
    print(f"{'Path B: MAE Embeddings (256-d)':<32}{bd_a[0]:<24}{bd_a[1]['pooled_eer']:>8.4f}{bd_a[1]['pooled_auc']:>8.4f}")
    if res_twin:
        print(f"{'Path C: VAE/KL Personal Twin':<32}{'PersonalHeartTwin (KL)':<24}{res_twin['mean_per_user_eer']:>8.4f}{res_twin['mean_per_user_auc']:>8.4f} (mean)")
        print(f"{'Path C: VAE/KL (Pooled)':<32}{'PersonalHeartTwin (KL)':<24}{res_twin['pooled_eer']:>8.4f}{res_twin['pooled_auc']:>8.4f} (pool)")
    print("Top features:", ", ".join(f['feature'] for f in feat_imp[:6]))
    print("=======================================================================================")
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
