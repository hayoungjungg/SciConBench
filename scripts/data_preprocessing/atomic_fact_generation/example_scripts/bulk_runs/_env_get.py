#!/usr/bin/env python3
"""Print a single value from a .env file, parsed with python-dotenv.

Used by the run_<model>.sh wrapper scripts instead of `source .env`, because
`.env` files are allowed syntax (e.g. spaces around '=', quoted values) that
python-dotenv tolerates but a raw bash `source` does not.

Usage: python3 _env_get.py <path-to-env> <KEY>

Prints the value and exits 0 if the key is present, otherwise prints nothing
and exits 1.
"""
import sys

from dotenv import dotenv_values


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: _env_get.py <env-path> <key>", file=sys.stderr)
        sys.exit(2)

    env_path, key = sys.argv[1], sys.argv[2]
    value = dotenv_values(env_path).get(key)
    if value is None:
        sys.exit(1)
    print(value)


if __name__ == "__main__":
    main()
