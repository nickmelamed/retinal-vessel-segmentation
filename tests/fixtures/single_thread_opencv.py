"""Pytest plugin that turns off OpenCV's thread pool, for mutation testing.

mutmut runs each mutant's tests in a process forked from one that has
already used OpenCV. On macOS, a forked child that touches OpenCV's thread
pool dies with SIGSEGV, which mutmut reports as "segfault" rather than a
test result. Loading this plugin before any test runs keeps the pool from
starting. ``[tool.mutmut]`` in pyproject.toml loads it with ``-p``.
"""

import cv2

cv2.setNumThreads(0)
