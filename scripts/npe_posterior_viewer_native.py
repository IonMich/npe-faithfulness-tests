from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

import npe_posterior_viewer as viewer_app


DEFAULT_ICON = Path("/Users/ioannism/Desktop/NPE Posterior Viewer.app/Contents/Resources/icon.icns")


def viewer_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/"


def api_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}api/models", timeout=0.75) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError):
        return False


def set_dock_icon(icon_path: Path) -> None:
    if not icon_path.exists():
        return
    try:
        from AppKit import NSApplication, NSImage

        app = NSApplication.sharedApplication()
        image = NSImage.alloc().initByReferencingFile_(str(icon_path))
        app.setApplicationIconImage_(image)
    except Exception as exc:  # noqa: BLE001 - best-effort macOS polish.
        print(f"Could not set Dock icon: {exc}", file=sys.stderr)


def start_internal_server(args: argparse.Namespace) -> viewer_app.ThreadingHTTPServer:
    viewer = viewer_app.NPEPosteriorViewer(
        args.model,
        args.broad_model,
        args.best_broad_model,
        args.best_broad_spline_model,
        args.best_broad_efficiency_model,
        args.best_broad_ensemble_summary,
        args.weighted_broad_ensemble_summary,
        seed=args.seed,
        device=args.device,
        mcmc_device=args.mcmc_device,
        mcmc_chains=args.mcmc_chains,
        mcmc_steps=args.mcmc_steps,
        mcmc_burn_in=args.mcmc_burn_in,
        mcmc_proposal_scale=args.mcmc_proposal_scale,
    )
    handler = viewer_app.make_handler(viewer, ui_dist=args.ui_dist.resolve())
    server, actual_port = viewer_app.make_server(
        host=args.host,
        port=args.port,
        handler=handler,
        port_retries=0,
        strict_port=True,
    )
    if actual_port != args.port:
        raise RuntimeError(f"Expected port {args.port}, got {actual_port}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the NPE posterior viewer in a native window.")
    parser.add_argument("--model", type=Path, default=viewer_app.DEFAULT_MODEL)
    parser.add_argument("--broad-model", type=Path, default=viewer_app.DEFAULT_BROAD_MODEL)
    parser.add_argument("--best-broad-model", type=Path, default=viewer_app.DEFAULT_BEST_BROAD_MODEL)
    parser.add_argument(
        "--best-broad-spline-model",
        type=Path,
        default=viewer_app.DEFAULT_BEST_BROAD_SPLINE_MODEL,
    )
    parser.add_argument(
        "--best-broad-efficiency-model",
        type=Path,
        default=viewer_app.DEFAULT_BEST_BROAD_EFFICIENCY_MODEL,
    )
    parser.add_argument(
        "--best-broad-ensemble-summary",
        type=Path,
        default=viewer_app.DEFAULT_BEST_BROAD_ENSEMBLE_SUMMARY,
    )
    parser.add_argument(
        "--weighted-broad-ensemble-summary",
        type=Path,
        default=viewer_app.DEFAULT_WEIGHTED_BROAD_ENSEMBLE_SUMMARY,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=viewer_app.DEFAULT_PORT)
    parser.add_argument("--ui-dist", type=Path, default=viewer_app.DEFAULT_UI_DIST)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--mcmc-device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--mcmc-chains", type=int, default=8)
    parser.add_argument("--mcmc-steps", type=int, default=24_000)
    parser.add_argument("--mcmc-burn-in", type=int, default=6_000)
    parser.add_argument(
        "--mcmc-proposal-scale",
        type=viewer_app.parse_proposal_scale,
        default=(0.030, 0.030, 0.040),
    )
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--icon", type=Path, default=Path(os.environ.get("NPE_VIEWER_ICON", DEFAULT_ICON)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = viewer_url(args.host, args.port)
    server: viewer_app.ThreadingHTTPServer | None = None

    if not api_ready(url):
        server = start_internal_server(args)
        for _ in range(80):
            if api_ready(url):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"NPE posterior viewer did not become ready at {url}")

    window = webview.create_window(
        "NPE Posterior Viewer",
        url,
        width=args.width,
        height=args.height,
        min_size=(980, 680),
        background_color="#f5f7fb",
        easy_drag=False,
    )

    def on_loaded() -> None:
        set_dock_icon(args.icon)
        window.load_css("body { -webkit-user-select: text !important; user-select: text !important; }")

    window.events.loaded += on_loaded

    try:
        webview.start(debug=args.debug)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
