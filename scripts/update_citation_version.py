#!/usr/bin/env python3
"""
Script to automatically update the version in CITATION.cff to match pyproject.toml
"""
import sys
import re
from pathlib import Path


def get_version_from_pyproject(pyproject_path):
    """Extract version from pyproject.toml"""
    try:
        with open(pyproject_path, "r") as f:
            content = f.read()

        # Find version in pyproject.toml
        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            return version_match.group(1)
        else:
            print("Error: Could not find version in pyproject.toml")
            return None
    except FileNotFoundError:
        print(f"Error: Could not find {pyproject_path}")
        return None


def update_citation_version(citation_path, new_version):
    """Update version in CITATION.cff"""
    try:
        with open(citation_path, "r") as f:
            content = f.read()

        # Replace version in CITATION.cff
        updated_content = re.sub(
            r"version:\s*[^\n]+", f"version: {new_version}", content
        )

        # Check if version was actually updated
        if content != updated_content:
            with open(citation_path, "w") as f:
                f.write(updated_content)
            print(f"Updated CITATION.cff version to {new_version}")
            return True
        else:
            print("CITATION.cff version is already up to date")
            return False

    except FileNotFoundError:
        print(f"Error: Could not find {citation_path}")
        return False


def main():
    """Main function"""
    # Get the repository root (assuming script is in scripts/ subdirectory)
    repo_root = Path(__file__).parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    citation_path = repo_root / "CITATION.cff"

    # Get version from pyproject.toml
    version = get_version_from_pyproject(pyproject_path)
    if not version:
        sys.exit(1)

    # Update CITATION.cff
    updated = update_citation_version(citation_path, version)

    # Exit with code 1 if files were modified (for pre-commit to stage changes)
    if updated:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
