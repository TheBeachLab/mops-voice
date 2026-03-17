"""Entry point for python -m mops_voice."""

import asyncio
from mops_voice.main import run


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
