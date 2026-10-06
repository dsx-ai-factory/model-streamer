# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
import typing
import unittest

import runai_model_streamer


class TestAnnotationsResolve(unittest.TestCase):
    """Every annotation in the package must actually resolve.

    These modules use `from __future__ import annotations`, which stores annotations as STRINGS and never
    evaluates them. That hides two classes of bug from the test suite and from import itself:

      - a type used in an annotation but never imported (`Tuple` without `from typing import Tuple`);
      - syntax the target interpreter cannot evaluate, e.g. PEP 604 `str | None` on Python 3.8.

    Neither shows up when the code runs. They surface only when something resolves the annotations - a type
    checker, `typing.get_type_hints`, a docs generator, pydantic - or the moment someone deletes the
    `from __future__ import annotations` line, at which point the module fails at IMPORT.

    Both bugs were present in this package and invisible until a reviewer read them. This test resolves
    every annotation so the next one fails here instead.
    """

    # Modules allowed to fail to import, mapped to why. Empty on purpose: every module in the package
    # imports cleanly in the environment the suite runs in. An entry here means "this module is not
    # annotation-checked", so adding one should be a conscious trade, not a side effect of a broad except.
    OPTIONAL_MODULES = {}

    def _import_package_modules(self):
        """Import every module in the package. Returns (modules, failures).

        A module that fails to import is a FAILURE, not something to skip: it is never annotation-checked,
        so tolerating it would let this test quietly stop covering whatever broke - which is the same
        silent-degradation the test exists to prevent. Every module in the package imports cleanly today,
        so there is nothing to tolerate.

        If a genuinely optional dependency is introduced later, add the module to OPTIONAL_MODULES with a
        reason. That is a deliberate, reviewable decision; broadening the except clause is not.
        """
        modules = []
        failures = []

        # walk_packages swallows ImportError from a package's __init__ unless onerror is given, which
        # would silently drop that package AND every module under it.
        def onerror(name):
            failures.append(f"{name}: package failed to import while walking (its submodules were skipped)")

        for module_info in pkgutil.walk_packages(
            runai_model_streamer.__path__, prefix=f"{runai_model_streamer.__name__}.", onerror=onerror
        ):
            name = module_info.name
            if ".tests" in name:
                continue
            try:
                modules.append(importlib.import_module(name))
            except Exception as error:
                if name in self.OPTIONAL_MODULES:
                    continue
                failures.append(f"{name}: {type(error).__name__}: {error}")

        return modules, failures

    def _annotated_targets(self, module):
        """Functions and classes DEFINED in this module, plus the classes' own methods."""
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            # __module__ filters out names merely imported into this module - those belong to whichever
            # module defines them, and are checked when that module is swept.
            if getattr(obj, "__module__", None) != module.__name__:
                continue

            if callable(obj) or isinstance(obj, type):
                yield attr_name, obj

            if isinstance(obj, type):
                for member_name in vars(obj):
                    member = getattr(obj, member_name, None)
                    if callable(member):
                        yield f"{attr_name}.{member_name}", member

    def test_every_module_imports(self):
        """A module that cannot be imported is never annotation-checked, so guard that separately.

        Asserted in its own test so a broken import is reported as a broken import, rather than as a
        missing annotation failure or - worse - as a pass.
        """
        modules, import_failures = self._import_package_modules()

        self.assertEqual(
            import_failures,
            [],
            "modules in the package failed to import, so their annotations were never checked:\n  "
            + "\n  ".join(import_failures),
        )
        self.assertTrue(modules, "no modules were discovered in the package - the walk found nothing")

    def test_every_annotation_resolves(self):
        modules, import_failures = self._import_package_modules()
        self.assertEqual(import_failures, [], "see test_every_module_imports")
        self.assertTrue(modules, "no modules were discovered in the package - the walk found nothing")

        failures = []
        for module in modules:
            for label, target in self._annotated_targets(module):
                try:
                    typing.get_type_hints(target)
                except Exception as error:
                    failures.append(f"{module.__name__}.{label}: {type(error).__name__}: {error}")

        self.assertEqual(
            failures,
            [],
            "annotations that do not resolve - a missing import, or syntax this interpreter cannot "
            "evaluate. They are silent today only because of `from __future__ import annotations`, and "
            "would become import errors without it:\n  " + "\n  ".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
