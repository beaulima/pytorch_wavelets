#!/usr/bin/env python
"""Regenerate notebooks/ from the gallery scripts in examples/.

The scripts under ``examples/`` are the source of truth: sphinx-gallery runs
them when the documentation is built, so they cannot quietly stop working. The
notebooks here are a convenience copy for people browsing the repository or
opening a demo in Colab, and are generated from those scripts rather than
edited by hand.

Run ``make notebooks`` after changing an example. CI checks that the two are in
step and fails if they are not.
"""
import argparse
import json
import pathlib
import sys

from sphinx_gallery.notebook import python_to_jupyter_cli

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / 'examples'
NOTEBOOKS = ROOT / 'notebooks'

# sphinx-gallery writes a kernelspec named 'python3', which resolves to
# whichever python3 kernel the running Jupyter happens to find first - often
# an unrelated environment, and then the first cell fails on the imports.
# Naming the kernel explicitly makes opening a notebook select the right one.
# `make kernel` registers exactly this name.
KERNELSPEC = {
    'display_name': 'Python (pytorch_wavelets)',
    'language': 'python',
    'name': 'pytorch_wavelets',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--check', action='store_true',
        help='exit non-zero if any notebook is missing or out of date, '
             'without writing anything')
    args = parser.parse_args(argv)

    scripts = sorted(EXAMPLES.glob('plot_*.py'))
    if not scripts:
        sys.exit('no examples/plot_*.py found')

    NOTEBOOKS.mkdir(exist_ok=True)
    stale = []
    for script in scripts:
        # The converter writes alongside its input, so build in examples/ and
        # move the result over.
        python_to_jupyter_cli([str(script)])
        produced = script.with_suffix('.ipynb')
        target = NOTEBOOKS / produced.name
        nb = json.loads(produced.read_text())
        nb['metadata']['kernelspec'] = KERNELSPEC
        # Canonicalise: the converter does not guarantee a stable key order
        # between runs, and comparing raw bytes would report that as drift.
        new = json.dumps(nb, indent=1, sort_keys=True) + '\n'
        produced.unlink()

        if args.check:
            current = (json.dumps(json.loads(target.read_text()), indent=1,
                                  sort_keys=True) + '\n'
                       if target.exists() else None)
            if current != new:
                stale.append(target.name)
        else:
            target.write_text(new)
            print('wrote notebooks/%s' % target.name)

    if args.check:
        if stale:
            sys.exit('out of date with examples/: %s\nrun `make notebooks`'
                     % ', '.join(stale))
        check_first_cell_is_self_contained()
        print('notebooks/ is in step with examples/ (%d files)' % len(scripts))


def check_first_cell_is_self_contained():
    """The first code cell must stand on its own.

    Notebooks get run out of order - that is the whole point of them - so the
    cell that sets up the environment has to carry its own imports. Keeping
    them in a separate earlier cell means anyone who starts from the seeding
    cell gets a NameError.
    """
    bad = []
    for path in sorted(NOTEBOOKS.glob('plot_*.ipynb')):
        cells = json.loads(path.read_text())['cells']
        code = [c for c in cells if c['cell_type'] == 'code']
        first = ''.join(code[0]['source']) if code else ''
        if 'import torch' not in first or 'SEED' not in first:
            bad.append(path.name)
    if bad:
        sys.exit('first code cell does not import what it uses: %s\n'
                 'keep the imports in the same cell as the seed'
                 % ', '.join(bad))
    check_no_execution_residue()


def check_no_execution_residue():
    """Committed notebooks must carry no trace of having been run.

    Running one in place - to check that it still works, say - writes
    ExecuteTime metadata, execution counts and outputs back into the file.
    That is machine-specific noise, it makes the diffs unreadable, and it is
    easy to commit without noticing.
    """
    dirty = []
    for path in sorted(NOTEBOOKS.glob('plot_*.ipynb')):
        for cell in json.loads(path.read_text())['cells']:
            if (cell.get('outputs') or cell.get('execution_count')
                    or 'ExecuteTime' in cell.get('metadata', {})):
                dirty.append(path.name)
                break
    if dirty:
        sys.exit('carries execution output or timings: %s\n'
                 'run `make notebooks` to regenerate them clean'
                 % ', '.join(dirty))


if __name__ == '__main__':
    main()
