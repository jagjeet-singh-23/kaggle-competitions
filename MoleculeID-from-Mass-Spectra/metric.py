"""MRR@25 exactly as the competition scores it.

A prediction is correct when it has the same atom connectivity as the answer. Both
sides go through RDKit tautomer canonicalisation (pinned at 2026.03.3) and are cut
down to the first block of the InChIKey, which encodes the 2D skeleton and drops
stereochemistry. So getting stereocentres or tautomer forms wrong costs nothing,
and the search space is smaller than it first looks.

    python3 metric.py      # self-check against the examples in the rules
"""
import functools

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
K = 25

_enum = rdMolStandardize.TautomerEnumerator()


@functools.lru_cache(maxsize=1_000_000)
def key14(smiles):
    """Canonical InChIKey first block, or None if the SMILES will not parse.

    Cached because a ranked list of 25 candidates for 400 molecules re-derives the
    same common scaffolds constantly, and canonicalisation is the expensive step.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _enum.Canonicalize(mol)
    except Exception:
        pass            # fall back to the untautomerised molecule rather than drop it
    k = Chem.MolToInchiKey(mol)
    return k[:14] if k else None


def reciprocal_rank(pred_smiles, answer_smiles):
    """1/rank of the first candidate matching the answer, else 0."""
    target = key14(answer_smiles)
    if target is None:
        return 0.0
    for i, s in enumerate(pred_smiles[:K], start=1):
        if key14(s) == target:
            return 1.0 / i
    return 0.0


def mrr(preds, answers):
    """preds: {molecule_id: [smiles, ...]}, answers: {molecule_id: smiles}."""
    if not answers:
        return 0.0
    return sum(reciprocal_rank(preds.get(m, []), a) for m, a in answers.items()) / len(answers)


def validate(df):
    """The submission rules, checked locally so a bad file is caught before upload."""
    assert {"molecule_id", "smiles"} <= set(df.columns), "missing required column"
    assert len(df), "empty submission"
    assert df.molecule_id.notna().all() and df.smiles.notna().all(), "nulls present"
    assert not df.molecule_id.duplicated().any(), "duplicate molecule_id"
    n = df.smiles.str.split(";").str.len()
    assert n.max() <= K, f"{int(n.max())} guesses for one molecule, limit is {K}"
    return True


def main():
    # Both spellings of glucose reduce to one key; this is the rules' own example.
    a = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
    b = "OCC1OC(O)C(O)C(O)C1O"
    assert key14(a) == key14(b) == "WQZGKKKJIJFFOK", (key14(a), key14(b))

    # Rank position is what scores, and only the first hit counts.
    assert reciprocal_rank([b], a) == 1.0
    assert reciprocal_rank(["CCO", b], a) == 0.5
    assert reciprocal_rank(["CCO"] * 24 + [b], a) == 1 / 25
    assert reciprocal_rank(["CCO"] * 25 + [b], a) == 0.0, "past rank 25 must not count"
    assert reciprocal_rank(["not a smiles", b], a) == 0.5, "junk should not crash"
    assert reciprocal_rank([], a) == 0.0

    # Tautomers must collide: the keto and enol forms of acetylacetone.
    assert key14("CC(=O)CC(C)=O") == key14("CC(O)=CC(C)=O")

    assert mrr({"m1": [b], "m2": ["CCO"]}, {"m1": a, "m2": a}) == 0.5
    print("metric self-checks passed")

    import time
    t = time.time()
    for s in ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "c1ccccc1"] * 100:
        key14.__wrapped__(s)
    print(f"key14 uncached: {(time.time() - t) / 300 * 1000:.1f} ms per SMILES")


if __name__ == "__main__":
    main()
