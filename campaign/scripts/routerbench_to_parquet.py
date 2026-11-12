"""Convert RouterBench's routerbench_0shot.pkl to the parquet file the
RouterBench analysis reads (scripts/routerbench_eval.py).

RouterBench (withmartian/routerbench on Hugging Face) is published only as a
pickle. A pickle can run arbitrary code when loaded, so this script never calls
pickle.load or pandas.read_pickle directly. It uses an allowlist unpickler that
only resolves the pandas and NumPy classes a DataFrame needs, and refuses
anything else (os, subprocess, eval, file readers, ...).

Usage, from campaign/:
    python scripts/routerbench_to_parquet.py path/to/routerbench_0shot.pkl

Writes ../external/routerbench/routerbench_0shot.parquet and prints the
SHA-256 of the input file so the download can be checked against the record.
"""
import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "external" / "routerbench" / "routerbench_0shot.parquet"

# Modules a pickled DataFrame legitimately references.
ALLOWED_PREFIXES = (
    "pandas.core.frame", "pandas.core.series", "pandas.core.internals",
    "pandas.core.indexes", "pandas.core.arrays", "pandas.core.dtypes",
    "pandas._libs", "pandas.core.generic",
    "numpy", "numpy.core", "numpy._core",
)
ALLOWED_EXACT = {
    ("builtins", "slice"), ("builtins", "object"), ("builtins", "complex"),
    ("copyreg", "_reconstructor"), ("_codecs", "encode"),
    ("datetime", "datetime"), ("datetime", "timedelta"), ("datetime", "date"),
    ("functools", "partial"),
}
# Never resolve these names, even inside an allowed module.
BLOCKED_WORDS = ("eval", "exec", "load", "read", "open", "system", "compile",
                 "import", "getattr", "f2py", "testing", "ctypes")


class AllowlistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(w in name.lower() for w in BLOCKED_WORDS) or \
           any(w in module.lower() for w in BLOCKED_WORDS):
            raise pickle.UnpicklingError("blocked: %s.%s" % (module, name))
        if (module, name) in ALLOWED_EXACT or module.startswith(ALLOWED_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError("not on the allowlist: %s.%s" % (module, name))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(prog="routerbench_to_parquet")
    ap.add_argument("pkl", type=Path, help="routerbench_0shot.pkl from withmartian/routerbench")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    print("input sha256:", sha256(a.pkl))
    with open(a.pkl, "rb") as f:
        df = AllowlistUnpickler(f).load()
    if not isinstance(df, pd.DataFrame):
        print("expected a pandas DataFrame, got %s" % type(df).__name__, file=sys.stderr)
        return 1
    for col in ("sample_id", "eval_name"):
        if col not in df.columns:
            print("missing expected column: %s" % col, file=sys.stderr)
            return 1

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print("wrote %s  (%d rows, %d columns)" % (a.out, len(df), df.shape[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
