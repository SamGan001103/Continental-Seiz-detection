"""The repo directory named `packaging/` collides with the PyPI package.

`packaging` is a real, very widely installed distribution — TensorFlow, pip and
setuptools all depend on it. This repository also has a top-level directory of
that name holding the build scripts, and it has no `__init__.py`, so it is a
namespace package. The consequence is that one import statement resolves to two
different things:

    legacy stack (3.6)   import packaging -> ./packaging        (this repo)
    modern stack (3.12)  import packaging -> site-packages      (the real one)

Nothing in the application imports it, which is the only reason this is
currently harmless. These tests keep it that way.

Adding `__init__.py` would NOT be the fix: it would make this directory a
regular package that shadows the real `packaging` whenever the repo is on
sys.path, which is exactly when TensorFlow is being imported. Renaming the
directory would be the real fix, and it touches the spec, the build scripts and
several documents — worth doing deliberately, not in passing.

The build scripts are unaffected because they run `packaging/smoke_test.py` as a
path, never as a module.
"""
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class NothingImportsTheAmbiguousName(unittest.TestCase):
    def test_no_python_file_imports_packaging_as_a_module(self):
        """An import that means different things on two stacks is a trap."""
        pattern = re.compile(r'^\s*(?:import\s+packaging\b'
                             r'|from\s+packaging(?:\.|\s+import))', re.M)
        offenders = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs
                       if d not in ('.git', '__pycache__', 'sample_data',
                                    'dist', 'build', 'artifacts',
                                    'thesis_report_bundle', 'USB_STAGING')]
            for fn in files:
                if not fn.endswith('.py') or fn.startswith('_'):
                    continue
                p = os.path.join(root, fn)
                try:
                    with open(p, encoding='utf-8', errors='replace') as f:
                        src = f.read()
                except OSError:
                    continue
                if pattern.search(src):
                    offenders.append(os.path.relpath(p, REPO))
        self.assertEqual(
            offenders, [],
            'these files import the name `packaging`, which is this repo on '
            'the 3.6 stack and the PyPI package on the modern one: {}. Load '
            'the build scripts by path instead.'.format(offenders))

    def test_the_directory_still_has_no_init(self):
        """__init__.py here would shadow the real packaging for TensorFlow."""
        self.assertFalse(
            os.path.exists(os.path.join(REPO, 'packaging', '__init__.py')),
            'packaging/__init__.py makes this directory shadow the PyPI '
            '`packaging` distribution whenever the repo is on sys.path — '
            'including while TensorFlow is importing. Rename the directory '
            'instead if it needs to be a package.')

    def test_the_build_scripts_are_still_reachable_by_path(self):
        """However the name resolves, running the file must work."""
        out = subprocess.run(
            [sys.executable, os.path.join(REPO, 'packaging', 'smoke_test.py'),
             '--help'],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=300).stdout.decode('utf-8', 'replace')
        self.assertIn('--dist', out)


if __name__ == '__main__':
    unittest.main()
