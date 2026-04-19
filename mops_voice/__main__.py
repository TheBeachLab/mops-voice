"""Entry point for `python -m mops_voice`. The console-script entry
point `mops-voice` (registered in pyproject.toml) calls the same main."""

from mops_voice.main import main

if __name__ == "__main__":
    main()
