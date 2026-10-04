#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco_evaluator import COCOEvaluator
try:
    from .mot_evaluator import MOTEvaluator
except ImportError:  # tracker-only dependencies (e.g. filterpy) are not needed for detector training
    MOTEvaluator = None
