import sys

from exp.shared.pipeline import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "features"]
    main()
