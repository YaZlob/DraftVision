from mlcore.core import Task
from ._solver import BaseSolver
from .segmentation import SegmentationSolver
from .detection import DetectionSolver

SOLVER: dict[str:BaseSolver] = {
    Task.SEMANTIC_SEGMENTATION: SegmentationSolver,
    Task.OBJECT_DETECTION: DetectionSolver,
}
