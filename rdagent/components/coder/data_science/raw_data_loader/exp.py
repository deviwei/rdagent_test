import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class DataLoaderTask(CoSTEERTask):
    pass
