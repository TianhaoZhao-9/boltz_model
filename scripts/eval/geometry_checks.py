"""
Physical geometry checks for predicted structures (6 checks):
  1. bond_lengths    - backbone bond lengths
  2. bond_angles     - backbone bond angles
  3. internal_clash  - VDW clashes within-chain
  4. chirality       - L-amino acid chirality
  5. stereo_bond     - peptide bond planarity (omega angle)
  6. interchain_clash- VDW clashes between chains
"""

import numpy as np
import gemmi


# Reference bond lengths (Angstrom) for backbone atoms WITHIN a residue
BB_BOND_LENGTHS = {
    ("N", "CA"): (1.46, 0.05),
    ("CA", "C"): (1.52, 0.05),
    ("C", "O"): (1.23, 0.05),   # carbonyl C=O
}
# Peptide bond C→N checked separately across consecutive residues
PEPTIDE_BOND = (1.33, 0.05)

# VDW radii by element (Angstrom)
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
VDW_CLASH_BUFFER = 0.85  # fraction of sum of radii below which is a clash


def load_structure(cif_path: str) -> gemmi.Structure:
    doc = gemmi.cif.read(cif_path)
    return gemmi.make_structure_from_block(doc[-1])


def get_atoms(model: gemmi.Model):
    """Return list of (chain_idx, atom_element, xyz) for all non-H atoms."""
    atoms = []
    for chain_idx, chain in enumerate(model):
        for res in chain:
            for atom in res:
                if atom.element.name not in ("H", "D", "X"):
                    atoms.append({
                        "chain_idx": chain_idx,
                        "chain_id": chain.name,
                        "res_name": res.name,
                        "atom_name": atom.name,
                        "element": atom.element.name,
                        "xyz": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                    })
    return atoms


def check_bond_lengths(model: gemmi.Model, buffer: float = 0.125) -> dict:
    """Check backbone bond lengths against ideal values."""
    violations = 0
    total = 0
    for chain in model:
        residues = list(chain)
        for i, res in enumerate(residues):
            atom_dict = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
            for (a1, a2), (ideal, tol) in BB_BOND_LENGTHS.items():
                if a1 in atom_dict and a2 in atom_dict:
                    dist = float(np.linalg.norm(atom_dict[a1] - atom_dict[a2]))
                    total += 1
                    lo, hi = ideal * (1 - buffer), ideal * (1 + buffer)
                    if dist < lo or dist > hi:
                        violations += 1
            # Peptide bond C(i) → N(i+1): only count if plausibly covalent (<2.5 Å)
            if i + 1 < len(residues):
                next_res = residues[i + 1]
                next_dict = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in next_res}
                if "C" in atom_dict and "N" in next_dict:
                    dist = float(np.linalg.norm(atom_dict["C"] - next_dict["N"]))
                    if dist < 2.5:
                        ideal, _ = PEPTIDE_BOND
                        total += 1
                        lo, hi = ideal * (1 - buffer), ideal * (1 + buffer)
                        if dist < lo or dist > hi:
                            violations += 1
    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.05, "violations": violations, "total": total, "viol_rate": rate}


def check_bond_angles(model: gemmi.Model, buffer: float = 0.125) -> dict:
    """Check backbone N-CA-C angles against ideal ~111 degrees."""
    IDEAL_NCA_C = 111.2
    IDEAL_CAC_N = 116.2
    IDEAL_CN_CA = 121.7
    violations = 0
    total = 0

    def angle_deg(a, b, c):
        v1, v2 = a - b, c - b
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    for chain in model:
        residues = list(chain)
        for i, res in enumerate(residues):
            atom_dict = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
            if all(k in atom_dict for k in ("N", "CA", "C")):
                ang = angle_deg(atom_dict["N"], atom_dict["CA"], atom_dict["C"])
                total += 1
                if abs(ang - IDEAL_NCA_C) > IDEAL_NCA_C * buffer:
                    violations += 1

                ang2 = angle_deg(atom_dict["CA"], atom_dict["C"], atom_dict.get("O", atom_dict["CA"]))
                if "O" in atom_dict:
                    total += 1
                    if abs(ang2 - IDEAL_CAC_N) > IDEAL_CAC_N * buffer:
                        violations += 1

    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.05, "violations": violations, "total": total, "viol_rate": rate}


def check_internal_clash(model: gemmi.Model, buffer: float = 0.225) -> dict:
    """Check for VDW clashes within each chain (non-bonded atom pairs)."""
    violations = 0
    total = 0
    for chain in model:
        atoms_xyz = []
        atoms_elem = []
        for res in chain:
            for atom in res:
                if atom.element.name not in ("H", "D", "X"):
                    atoms_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    atoms_elem.append(atom.element.name)
        if len(atoms_xyz) < 2:
            continue
        coords = np.array(atoms_xyz)
        # Check all pairs separated by >= 4 bonds (approximated by residue distance >= 2)
        # Quick approximation: check pairs that are > 4 in index but within 6A
        for i in range(len(coords)):
            for j in range(i + 4, min(i + 100, len(coords))):
                dist = np.linalg.norm(coords[i] - coords[j])
                r_i = VDW_RADII.get(atoms_elem[i], 1.7)
                r_j = VDW_RADII.get(atoms_elem[j], 1.7)
                min_dist = (r_i + r_j) * (1.0 - buffer)
                total += 1
                if dist < min_dist:
                    violations += 1
    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.02, "violations": violations, "total": total, "viol_rate": rate}


def check_chirality(model: gemmi.Model) -> dict:
    """Check L-amino acid chirality using CA-N-C-CB dihedral sign."""
    violations = 0
    total = 0
    GLYCINE = {"G", "GLY"}

    def dihedral(a, b, c, d):
        b1, b2, b3 = b - a, c - b, d - c
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return np.degrees(np.arctan2(y, x))

    for chain in model:
        for res in chain:
            if res.name in GLYCINE:
                continue
            atom_dict = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res}
            if all(k in atom_dict for k in ("N", "CA", "C", "CB")):
                total += 1
                phi = dihedral(atom_dict["N"], atom_dict["CA"], atom_dict["C"], atom_dict["CB"])
                # L-amino acids: CA chirality gives positive N-CA-C-CB dihedral
                if phi < 0:
                    violations += 1

    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.05, "violations": violations, "total": total, "viol_rate": rate}


def check_stereo_bond(model: gemmi.Model, buffer_deg: float = 30.0) -> dict:
    """Check peptide bond planarity (omega angle ~180 degrees)."""
    violations = 0
    total = 0

    def dihedral(a, b, c, d):
        b1, b2, b3 = b - a, c - b, d - c
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-8))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return np.degrees(np.arctan2(y, x))

    for chain in model:
        residues = list(chain)
        for i in range(len(residues) - 1):
            res1, res2 = residues[i], residues[i + 1]
            d1 = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res1}
            d2 = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res2}
            if all(k in d1 for k in ("CA", "C")) and all(k in d2 for k in ("N", "CA")):
                total += 1
                omega = dihedral(d1["CA"], d1["C"], d2["N"], d2["CA"])
                # Omega should be ~180 (trans) or ~0 (cis proline)
                if abs(abs(omega) - 180) > buffer_deg and abs(omega) > buffer_deg:
                    violations += 1

    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.05, "violations": violations, "total": total, "viol_rate": rate}


def check_interchain_clash(model: gemmi.Model, buffer: float = 0.225) -> dict:
    """Check VDW clashes between different chains."""
    violations = 0
    total = 0
    chains = list(model)
    if len(chains) < 2:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}

    # Collect chain atoms
    chain_atoms = []
    for chain in chains:
        atoms = []
        for res in chain:
            for atom in res:
                if atom.element.name not in ("H", "D", "X"):
                    atoms.append(([atom.pos.x, atom.pos.y, atom.pos.z], atom.element.name))
        chain_atoms.append(atoms)

    # Check inter-chain pairs (sample for speed)
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            coords_i = np.array([a[0] for a in chain_atoms[i]])
            elems_i = [a[1] for a in chain_atoms[i]]
            coords_j = np.array([a[0] for a in chain_atoms[j]])
            elems_j = [a[1] for a in chain_atoms[j]]
            if len(coords_i) == 0 or len(coords_j) == 0:
                continue
            # Sample atoms for speed (max 200 per chain)
            step_i = max(1, len(coords_i) // 200)
            step_j = max(1, len(coords_j) // 200)
            ci = coords_i[::step_i]
            cj = coords_j[::step_j]
            ei = elems_i[::step_i]
            ej = elems_j[::step_j]
            diffs = ci[:, None, :] - cj[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            for ii in range(len(ci)):
                for jj in range(len(cj)):
                    r_i = VDW_RADII.get(ei[ii], 1.7)
                    r_j = VDW_RADII.get(ej[jj], 1.7)
                    min_dist = (r_i + r_j) * (1.0 - buffer)
                    total += 1
                    if dists[ii, jj] < min_dist:
                        violations += 1

    if total == 0:
        return {"pass": True, "violations": 0, "total": 0, "viol_rate": 0.0}
    rate = violations / total
    return {"pass": rate < 0.02, "violations": violations, "total": total, "viol_rate": rate}


def evaluate_structure(cif_path: str) -> dict:
    """Run 6 geometry checks on a CIF structure file."""
    try:
        st = load_structure(cif_path)
        model = st[0]
        results = {
            "bond_lengths": check_bond_lengths(model),
            "bond_angles": check_bond_angles(model),
            "internal_clash": check_internal_clash(model),
            "chirality": check_chirality(model),
            "stereo_bond": check_stereo_bond(model),
            "interchain_clash": check_interchain_clash(model),
        }
        n_pass = sum(1 for v in results.values() if v["pass"])
        n_total = len(results)
        results["summary"] = {
            "checks_passed": n_pass,
            "total_checks": n_total,
            "pass_rate": n_pass / n_total,
        }
        return results
    except Exception as e:
        return {"error": str(e)}


_PROTEIN_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "ACE","NME","UNK","MSE",
}
_NUCLEIC_RESIDUES = {"A","G","C","U","DA","DG","DC","DT","ADE","GUA","CYT","URI","N","I"}

# The 6 PoseBusters binary-check columns that map to the spec's 6 items
_PB_CHECKS = [
    "bond_lengths",              # 键长
    "bond_angles",               # 键角
    "internal_steric_clash",     # 内部碰撞
    "passes_valence_checks",     # 手性（代理，无参考结构时最佳可用项）
    "double_bond_flatness",      # 键立体化学
    "minimum_distance_to_protein",  # 链间碰撞
]


def run_posebusters(cif_path: str) -> dict:
    """Run PoseBusters on the small-molecule ligand in a CIF file.

    Returns {"applicable": False} if no ligand chain is found.
    """
    import tempfile, os
    from rdkit import Chem
    from posebusters import PoseBusters

    try:
        st = load_structure(cif_path)
        model = st[0]

        ligand_ids, protein_ids = [], []
        for chain in model:
            names = {r.name for r in chain}
            if names <= _NUCLEIC_RESIDUES:
                continue
            if names <= _PROTEIN_RESIDUES:
                protein_ids.append(chain.name)
            else:
                ligand_ids.append(chain.name)

        if not ligand_ids:
            return {"applicable": False}

        # Write ligand and protein chains to temporary PDB files
        st_lig = st.clone()
        for cid in [c.name for c in model if c.name not in ligand_ids]:
            try:
                st_lig[0].remove_chain(cid)
            except Exception:
                pass

        st_prot = st.clone()
        for cid in ligand_ids:
            try:
                st_prot[0].remove_chain(cid)
            except Exception:
                pass

        lig_pdb = tempfile.mktemp(suffix=".pdb")
        prot_pdb = tempfile.mktemp(suffix=".pdb")
        st_lig.write_pdb(lig_pdb)
        st_prot.write_pdb(prot_pdb)

        mol = Chem.MolFromPDBFile(lig_pdb, removeHs=True, sanitize=True)
        if mol is None:
            return {"applicable": False, "error": "RDKit could not parse ligand"}

        sdf_path = tempfile.mktemp(suffix=".sdf")
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()

        pb = PoseBusters(config="dock")
        df = pb.bust(mol_pred=sdf_path, mol_cond=prot_pdb, full_report=True)

        checks = {}
        for col in _PB_CHECKS:
            if col in df.columns:
                checks[col] = bool(df[col].iloc[0])

        n_pass = sum(checks.values())
        n_total = len(_PB_CHECKS)

        return {
            "applicable": True,
            "checks": checks,
            "checks_passed": n_pass,
            "total_checks": n_total,
            "pass_rate": n_pass / n_total,
        }

    except Exception as e:
        return {"applicable": False, "error": str(e)}
    finally:
        for f in [locals().get("lig_pdb"), locals().get("prot_pdb"), locals().get("sdf_path")]:
            if f and os.path.exists(f):
                os.unlink(f)
