import sys

from exp.shared.run_pipeline import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "analysis"]
    main()
