#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .ava_dataset import Ava  # noqa
from .build import DATASET_REGISTRY, build_dataset  # noqa
from .charades import Charades  # noqa
from .kinetics import Kinetics  # noqa
from .ssv2 import Ssv2  # noqa
from .clevrer import Clevrerframe 
from .clevrer import Clevrer 
from .clevrer_dual import Clevrer_des 
from .slot_clevrer import Clevrerslot
from .clevrer_text import Clevrertext, Clevrertext_des, Clevrertext_mc, Clevrertext_join 