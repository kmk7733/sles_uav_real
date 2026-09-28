#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HPA reference producer; importing this package loads no ROS or torch.

The legacy simulator adapter remains hpa.policy.LearnedHPA. This package
extracts its physical-action/reference and commitment operations without
changing that adapter. The pinned V4 runtime is imported explicitly from
planner.hpa.v4 only when inference is required.
"""
from planner.hpa.reference import BodyActionChunk, actions_to_reference, world_actions
from planner.hpa.producer import HPAProducer
from planner.hpa.commit import commit_plan

__all__ = ["BodyActionChunk", "HPAProducer", "actions_to_reference",
           "world_actions", "commit_plan"]
