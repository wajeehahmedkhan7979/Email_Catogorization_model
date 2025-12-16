"""
Version metadata for the email categorisation worker.

Usage:
    from version import get_version
    print(get_version())
"""

__version__ = "v1.0.0"


def get_version() -> str:
    """
    Return the current package version string.
    """
    return __version__

