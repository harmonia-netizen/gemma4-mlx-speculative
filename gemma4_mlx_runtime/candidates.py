import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def __getattr__(name):
    if name == "CandidateRegistry":
        import sys
        import os
        experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
        if experiments_dir not in sys.path:
            sys.path.insert(0, experiments_dir)
        from experiments.template_draft_runtime import CandidateRegistry
        return CandidateRegistry
    raise AttributeError(name)

__all__ = ["CandidateRegistry"]
