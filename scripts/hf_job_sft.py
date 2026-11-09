"""Remote SFT entry. The launcher installs the repo, then runs this module."""

import sys

from schemasmith.hf.remote import main

if __name__ == "__main__":
    raise SystemExit(main(["sft", *sys.argv[1:]]))
