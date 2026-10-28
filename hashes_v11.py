"""SHA-256 of the version 1.1 files: python3 hashes_v11.py -> MUX-S1_v11.sha256"""
import hashlib, datetime
FILES = ['MUX-S1_v11_prereg.md', 'mux_s1/realdata.py', 'mux_s1/tests_v11.py', 'verify_real.py',
         'hashes_v11.py', 'mux_s1/adapters.py']
FROZEN = set(FILES) - {'mux_s1/adapters.py'}
lines = [f"# MUX-S1 version 1.1, hashed {datetime.datetime.now(datetime.UTC).isoformat()}",
         "# frozen: any change after publication is a deviation",
         "# mux_s1/adapters.py is NOT frozen: the DANDI reader is written on release day"]
for f in FILES:
    h = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    lines.append(f"{h}  {f}" + ("" if f in FROZEN else "   (not frozen)"))
open('MUX-S1_v11.sha256', 'w').write("\n".join(lines) + "\n")
print("\n".join(lines))
