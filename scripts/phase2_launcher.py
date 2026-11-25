"""Launcher wrapper for Phase 2 — handles module import + args."""
import sys, os

# Ensure project root is first in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if __name__ == "__main__":
    from fungidna.training.phase2_joint import train_phase2_joint
    train_phase2_joint(sys.argv[1])
