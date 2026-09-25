from pathlib import Path
import sys
WORKFLOW = Path(__file__).resolve().parents[1] / 'src' / 'cacao_image_benchmark' / 'workflow'
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))
