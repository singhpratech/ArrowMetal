"""Shared pytest setup.

The suites run small inputs that the router's `auto` mode would send to the CPU, so they pin the
router to the GPU and keep exercising the kernels. ARROWMETAL_ROUTER, when set, wins: running the
suites with ARROWMETAL_ROUTER=cpu exercises every CPU path against the same tests. test_router.py
chooses paths per call and runs gpu, cpu and auto. See docs/TESTING.md.
"""
import os

import arrowmetal as am

if "ARROWMETAL_ROUTER" not in os.environ:
    am.set_router("gpu")
