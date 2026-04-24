"""Evaluate baseline predictions: collect complex_pLDDT and run 6-item geometry checks."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from geometry_checks import evaluate_structure, run_posebusters

TARGETS = ["7ZGE", "8A1Q", "8C4I", "8D7M", "8GG5", "ligand", "cyclic_prot"]
TARGET_TYPES = {
    "7ZGE":        "protein-ligand",
    "8A1Q":        "homodimer",
    "8C4I":        "protein-RNA",
    "8D7M":        "monomer",
    "8GG5":        "protein-protein",
    "ligand":      "protein-ligand",
    "cyclic_prot": "cyclic-peptide",
}

# Each baseline lists directories in search order; first hit wins.
BASELINE_DIRS = {
    "baseline1_no_steering": [
        "d:/STAT435/boltz_model/results/baseline1/boltz_results_inputs",
        "d:/STAT435/boltz_model/results/baseline1/boltz_results_ligand/boltz_results_ligand",
        "d:/STAT435/boltz_model/results/baseline1/boltz_results_cyclic/boltz_results_cyclic_prot",
    ],
    "baseline2_full_steering": [
        "d:/STAT435/boltz_model/results/baseline2/boltz_results_inputs",
        "d:/STAT435/boltz_model/results/baseline2/boltz_results_ligand/boltz_results_ligand",
        "d:/STAT435/boltz_model/results/baseline2/boltz_results_cyclic/boltz_results_cyclic_prot",
    ],
}

CHECK_NAMES = ["bond_lengths", "bond_angles", "internal_clash", "chirality", "stereo_bond", "interchain_clash"]
CHECK_LABELS = {
    "bond_lengths":    "键长",
    "bond_angles":     "键角",
    "internal_clash":  "内部碰撞",
    "chirality":       "手性",
    "stereo_bond":     "肽键平面",
    "interchain_clash":"链间碰撞",
}


def get_confidence(pred_dir: Path, target: str) -> dict:
    conf_path = pred_dir / "predictions" / target / f"confidence_{target}_model_0.json"
    if not conf_path.exists():
        return {}
    with open(conf_path) as f:
        return json.load(f)


def find_pred_dir(dirs: list, target: str) -> Path | None:
    for d in dirs:
        p = Path(d) / "predictions" / target
        if p.exists():
            return Path(d)
    return None


def main():
    all_results = {}

    for bl_name, bl_dirs in BASELINE_DIRS.items():
        all_results[bl_name] = {}
        print(f"\nEvaluating {bl_name} ...")

        for target in TARGETS:
            pred_dir = find_pred_dir(bl_dirs, target)
            if pred_dir is None:
                print(f"  {target}: no predictions found, skipping")
                all_results[bl_name][target] = {"type": TARGET_TYPES[target], "missing": True}
                continue

            conf = get_confidence(pred_dir, target)
            cif_path = pred_dir / "predictions" / target / f"{target}_model_0.cif"
            print(f"  {target}: geometry...", end=" ", flush=True)
            geo = evaluate_structure(str(cif_path))
            print("PoseBusters...", end=" ", flush=True)
            pb = run_posebusters(str(cif_path))
            print("done")

            all_results[bl_name][target] = {
                "type": TARGET_TYPES[target],
                "complex_plddt": conf.get("complex_plddt"),
                "geometry": geo,
                "posebusters": pb,
            }

    # ── Summary Table (pLDDT + PoseBusters) ──────────────────────────────────
    print("\n" + "=" * 90)
    print("  complex_pLDDT and PoseBusters Pass Rate")
    print("=" * 90)
    print(f"{'TARGET':<8} {'TYPE':<16} {'BL1 pLDDT':>10} {'BL2 pLDDT':>10} {'BL1 PB':>10} {'BL2 PB':>10}")
    print("-" * 90)
    for target in TARGETS:
        r1 = all_results["baseline1_no_steering"][target]
        r2 = all_results["baseline2_full_steering"][target]
        v1 = r1.get("complex_plddt")
        v2 = r2.get("complex_plddt")
        s1 = f"{v1:.4f}" if v1 is not None else "N/A"
        s2 = f"{v2:.4f}" if v2 is not None else "N/A"
        pb1 = r1.get("posebusters", {})
        pb2 = r2.get("posebusters", {})
        p1 = f"{pb1['pass_rate']:.1%}" if pb1.get("applicable") else "N/A"
        p2 = f"{pb2['pass_rate']:.1%}" if pb2.get("applicable") else "N/A"
        print(f"{target:<8} {r1['type']:<16} {s1:>10} {s2:>10} {p1:>10} {p2:>10}")
    print("-" * 90)

    # ── Geometry Table ────────────────────────────────────────────────────────
    col_w = 10
    sep = "=" * (8 + 16 + col_w * len(CHECK_NAMES) + 12 + 4)
    print("\n" + sep)
    print("  6-item Physical Geometry Check Pass Rate")
    print(sep)
    header = f"{'TARGET':<8} {'TYPE':<16}"
    for cn in CHECK_NAMES:
        header += f" {CHECK_LABELS[cn]:>{col_w}}"
    header += f" {'PASS_RATE':>10}"
    print(header)

    for bl_name in BASELINE_DIRS:
        print(f"\n  [{bl_name}]")
        print("-" * len(sep))
        pass_counts = {cn: 0 for cn in CHECK_NAMES}
        n_targets = 0
        for target in TARGETS:
            r = all_results[bl_name][target]
            if r.get("missing"):
                print(f"  {target:<8} MISSING")
                continue
            geo = r["geometry"]
            if "error" in geo:
                print(f"  {target:<8} ERROR: {geo['error']}")
                continue
            n_targets += 1
            row = f"  {target:<6} {r['type']:<16}"
            n_pass = 0
            for cn in CHECK_NAMES:
                check = geo.get(cn, {})
                passed = check.get("pass", False)
                vr = check.get("viol_rate", float("nan"))
                cell = f"OK({vr:.1%})" if passed else f"FAIL({vr:.1%})"
                if passed:
                    n_pass += 1
                    pass_counts[cn] += 1
                row += f" {cell:>{col_w}}"
            summary = geo.get("summary", {})
            pr = summary.get("pass_rate", n_pass / len(CHECK_NAMES))
            row += f" {pr:>10.1%}"
            print(row)

        if n_targets:
            print(f"  {'[pass rate]':<24}", end="")
            for cn in CHECK_NAMES:
                rate = pass_counts[cn] / n_targets
                print(f" {rate:>{col_w}.0%}", end="")
            overall = sum(pass_counts.values()) / (len(CHECK_NAMES) * n_targets)
            print(f" {overall:>10.0%}")

    # ── Delta Table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Steering Effect (BL2 - BL1) on complex_pLDDT")
    print("=" * 70)
    print(f"{'TARGET':<8} {'TYPE':<16} {'BL1':>10} {'BL2':>10} {'Δ':>8}")
    print("-" * 70)
    for target in TARGETS:
        r1 = all_results["baseline1_no_steering"][target]
        r2 = all_results["baseline2_full_steering"][target]
        v1 = r1.get("complex_plddt")
        v2 = r2.get("complex_plddt")
        if v1 is not None and v2 is not None:
            delta = v2 - v1
            sign = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
            print(f"{target:<8} {r1['type']:<16} {v1:>10.4f} {v2:>10.4f} {delta:>+7.4f} {sign}")

    # Save JSON
    out_path = Path("d:/STAT435/boltz_model/results/baseline_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
