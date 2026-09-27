"""Explicit, one-time setup of the clean candidate execution image."""

import argparse
from datetime import datetime, timezone
import json
import platform

from .cli import create_run_directory, write_private
from .suites import canonical_json


def setup(app_name, sdk):
    app = sdk.App.lookup(app_name, create_if_missing=True)
    # No local source, reference answers, volumes, secrets, or model dependencies.
    image = sdk.Image.debian_slim(python_version="3.12")
    with sdk.enable_output():
        image.build(app)
    return {
        "schema_version": "0.1", "app_name": app_name,
        "sandbox_image_id": image.object_id,
        "image_recipe": {"base": "debian_slim", "python_series": "3.12",
                         "local_files": [], "extra_dependencies": []},
        "modal_sdk_version": sdk.__version__,
        "controller_python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "live_conformance_passed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="verifier-rl-evaluation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-cloud", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_cloud:
        parser.error("--allow-cloud is required: setup creates Modal resources")
    import modal
    directory = create_run_directory(args.out)
    write_private(directory / "requested_setup.json", canonical_json({"app_name": args.app}))
    try:
        result = setup(args.app, modal)
        write_private(directory / "setup.json", canonical_json(result))
    except Exception as exc:
        # Avoid echoing SDK errors that might contain auth material.
        write_private(directory / "setup_error.json", canonical_json({"error_type": type(exc).__name__}))
        print(f"Setup failed: {type(exc).__name__}. Check local Modal authentication and service access.")
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
