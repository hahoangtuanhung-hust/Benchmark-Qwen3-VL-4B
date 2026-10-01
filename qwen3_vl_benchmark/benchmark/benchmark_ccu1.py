"""CCU1 entry point; delegates to the common backend-neutral runner."""
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())

