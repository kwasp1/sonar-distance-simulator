"""Makes the repo root importable so tests can `from src... import ...`.

pytest's default import mode puts the test file's own directory on sys.path,
not the repo root — so `from src.pulse import ...` fails without this.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))