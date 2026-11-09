"""Remote DPO entry. Runs inside a job, after the SFT adapter exists."""

import sys

from schemasmith.hf.remote import main

if __name__ == "__main__":
    raise SystemExit(main(["dpo", *sys.argv[1:]]))
