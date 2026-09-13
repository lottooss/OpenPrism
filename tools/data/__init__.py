"""OpenPrism dataset, recording, synthetic generation, splitting, and reporting tooling."""

from tools.data.annotation_verifier import AnnotationVerifier, DatasetAuditReport
from tools.data.dataset_manifest import DatasetManifest, DatasetManifestBuilder
from tools.data.dataset_report import generate_validation_report
from tools.data.dataset_splitter import DatasetSplitter, DatasetSplitterConfig, SplitDistributionReport
from tools.data.recording_index import RecordingIndexer, RecordingSession
from tools.data.synthetic_generator import (
    FrameAnnotation,
    IgnoreRegion,
    SyntheticGeneratorConfig,
    SyntheticTargetGenerator,
    TargetAnnotation,
)

__all__ = [
    "AnnotationVerifier",
    "DatasetAuditReport",
    "DatasetManifest",
    "DatasetManifestBuilder",
    "DatasetSplitter",
    "DatasetSplitterConfig",
    "FrameAnnotation",
    "IgnoreRegion",
    "RecordingIndexer",
    "RecordingSession",
    "SplitDistributionReport",
    "SyntheticGeneratorConfig",
    "SyntheticTargetGenerator",
    "TargetAnnotation",
    "generate_validation_report",
]
