# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone version-string parsing helper.

This module intentionally has no third-party dependencies (in particular no
``torch``) so that it can be imported both at build time (from ``setup.py``,
loaded by file path) and at runtime (from ``torchao/__init__.py`` and
``torchao/utils.py``) without pulling in the rest of the package.
"""

import re


def parse_version(version_string):
    """
    Parse version string representing pre-release with -1

    Examples: "2.5.0.dev20240708+cu121" -> [2, 5, -1], "2.5.0" -> [2, 5, 0]
    """
    # Check for pre-release indicators
    is_prerelease = bool(re.search(r"(git|dev)", version_string))
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_string)
    if match:
        major, minor, patch = map(int, match.groups())
        if is_prerelease:
            patch = -1
        return [major, minor, patch]
    else:
        raise ValueError(f"Invalid version string format: {version_string}")
