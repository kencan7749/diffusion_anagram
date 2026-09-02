"""A browser view of `runs/`: lineage, archive coverage, round metrics, media.

Re-exports the application factory so `flask --app ava.webui run` works.
Flask is the only import; the data layer is pure Python plus numpy/yaml.
"""

from ava.webui.app import create_app

__all__ = ["create_app"]
