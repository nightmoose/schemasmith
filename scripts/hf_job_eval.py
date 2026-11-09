"""Remote held-out compare. Writes compare.json and can push it privately."""

import sys

from schemasmith.hf.remote import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
