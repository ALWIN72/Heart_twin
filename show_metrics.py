"""
show_metrics.py
Display all saved benchmark, model report, and comparison metrics directly in the terminal.
"""
import json
from pathlib import Path

def show_benchmarks():
    p = Path("checkpoints/benchmark_results.json")
    if not p.exists():
        print("[!] checkpoints/benchmark_results.json not found. Run the benchmark first.")
        return
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    print("\n" + "=" * 65)
    print("           DIGITAL HEART TWIN - HEADLINE BENCHMARKS")
    print("=" * 65)
    targets = d.get("targets", {})
    eer = d.get("EER")
    eer_pooled = d.get("EER_pooled")
    eer_ci = d.get("EER_95_CI")
    auc = d.get("AUC_ROC")
    lead = d.get("Lead_Time_days")
    far = d.get("False_Alarm_Rate")
    
    eer_str = f"{eer:.4f}" if isinstance(eer, (int, float)) and not str(eer) == "nan" else str(eer)
    auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) and not str(auc) == "nan" else str(auc)
    far_str = f"{far:.4f}" if isinstance(far, (int, float)) and not str(far) == "nan" else str(far)
    
    print(f"  EER (Mean Per-User)    : {eer_str:<8} (Target: {targets.get('EER', 'N/A')})")
    if eer_pooled is not None:
        pooled_str = f"{eer_pooled:.4f}" if isinstance(eer_pooled, (int, float)) else str(eer_pooled)
        print(f"  EER (Pooled)           : {pooled_str:<8}")
    if eer_ci and len(eer_ci) == 2:
        print(f"  EER 95% Bootstrap CI   : [{eer_ci[0]}, {eer_ci[1]}]")
    print(f"  AUC-ROC (Early Warning): {auc_str:<8} (Target: {targets.get('AUC_ROC', 'N/A')})")
    print(f"  Lead Time (days)       : {str(lead):<8} (Target: {targets.get('Lead_Time_days', 'N/A')})")
    print(f"  False Alarm Rate       : {far_str:<8} (Target: {targets.get('False_Alarm_Rate', 'N/A')})")
    print("=" * 65 + "\n")

def show_comparison():
    p = Path("checkpoints/compare_results.json")
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    print("=" * 65)
    print("   MODEL COMPARISON: CLASSICAL ML vs. DEEP LEARNING vs. HYBRID")
    print("=" * 65)
    c = d.get("classical", {})
    dl = d.get("deep_learning", {})
    h = d.get("hybrid", {})

    print(f"  {'Metric':<28}{'Classical':>10}{'Deep Learning':>14}{'Hybrid':>10}")
    c_ver = c.get("verification", {})
    dl_ver = dl.get("verification", {})
    h_ver = h.get("verification", {})
    print(f"  {'Verify EER':<28}{c_ver.get('eer', 0):>10.4f}{dl_ver.get('eer', 0):>14.4f}{h_ver.get('eer', 0):>10.4f}")
    print(f"  {'Verify AUC':<28}{c_ver.get('auc', 0):>10.4f}{dl_ver.get('auc', 0):>14.4f}{h_ver.get('auc', 0):>10.4f}")

    c_ano = c.get("anomaly", {})
    dl_ano = dl.get("anomaly", {})
    h_ano = h.get("anomaly", {})
    print(f"  {'Anomaly Pooled EER':<28}{c_ano.get('pooled_eer', 0):>10.4f}{dl_ano.get('pooled_eer', 0):>14.4f}{h_ano.get('pooled_eer', 0):>10.4f}")
    print(f"  {'Anomaly Mean/User EER':<28}{c_ano.get('mean_per_user_eer', 0):>10.4f}{dl_ano.get('mean_per_user_eer', 0):>14.4f}{h_ano.get('mean_per_user_eer', 0):>10.4f}")
    if "best_verification" in d:
        print(f"\n  Best Verification Model : {d['best_verification']}")
    if "best_anomaly" in d:
        print(f"  Best Anomaly Model      : {d['best_anomaly']}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    show_benchmarks()
    show_comparison()
