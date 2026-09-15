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
        # Canonicalise: the converter does not guarantee a stable key order
        # between runs, and comparing raw bytes would report that as drift.
        new = json.dumps(json.loads(produced.read_text()),
                         indent=1, sort_keys=True) + '\n'
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
        print('notebooks/ is in step with examples/ (%d files)' % len(scripts))


if __name__ == '__main__':
    main()
