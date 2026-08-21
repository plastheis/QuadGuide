import sys
from pathlib import Path

# tools/ is not an installed package; put it on sys.path so `import convert_lib` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
