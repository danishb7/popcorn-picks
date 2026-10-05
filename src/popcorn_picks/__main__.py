"""Allow commands to run with ``python -m popcorn_picks``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
