"""Allow running the audit module as ``python -m security.audit``."""

import sys

from .cli import main

sys.exit(main())
