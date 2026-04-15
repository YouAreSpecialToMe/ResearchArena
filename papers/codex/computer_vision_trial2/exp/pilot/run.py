import sys

from exp.shared.focus_pipeline import main


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "pilot"])
    main()
