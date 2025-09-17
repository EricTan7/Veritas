import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from utils.registry import DETECTOR
from .utils import slowfast

from .convnextv2_detector import ConvNextDetector
from .dinov2_detector import DINODetector
from .clip_detector import CLIPDetector
from .effort_detector import EffortDetector
from .freqnet_detector import FreqNetDetector
from .prodet_detector import ProDetDetector
from .iid_detector import IIDDetector
from .cospy_detector import COSPYDetector
from .d3_detector import D3Detector
from .npr_detector import NPRDetector
from .f3net_detector import F3netDetector
from .aeroblade_detector import AEROBLADEDetector
from .aide_detector import AIDEDetector
