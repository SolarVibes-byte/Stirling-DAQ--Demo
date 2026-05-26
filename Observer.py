# Observer.py
# Event-driven file watcher for the acquisition pipeline.
#
# The observer monitors an inbox directory. Whenever a new acquisition file is
# dropped in, it is processed by the MathematicalProcessor, its results are
# exported to CSV, and the source file is archived. Successful files go to a
# processed directory; files that raise an error go to a failed directory, so
# a single corrupt file never blocks the queue or is silently lost.
#
# The module separates three responsibilities:
#   process_and_archive  — pure logic (process, export, move). Directly testable.
#   InboxHandler         — watchdog event adapter.
#   watch_inbox          — starts the real observer loop.

from __future__ import annotations
import logging
import shutil
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer as _WatchdogObserver

import MathematicalProcessor as MP
import Exporter as Exp
import Visualization as Viz

logger = logging.getLogger(__name__)

# Default directory layout, all relative to the project root.
INBOX_DIRNAME     = "inbox"
PROCESSED_DIRNAME = "processed"
FAILED_DIRNAME    = "failed"
RESULTS_DIRNAME   = "results"

# A file is considered fully written once its size is unchanged across this
# many consecutive polls (separated by _STABLE_POLL_S seconds).
_STABLE_POLLS  = 3
_STABLE_POLL_S = 0.4


# ──────────────────────────────────────────────────────────────────────
#  1. FILE-READINESS GUARD
# ──────────────────────────────────────────────────────────────────────

def wait_until_stable(path: Path,
                      polls: int = _STABLE_POLLS,
                      interval_s: float = _STABLE_POLL_S,
                      timeout_s: float = 30.0,
                      ) -> bool:
    """Block until a file's size stops changing, so it is read only once fully
    written.

    A file-created event fires when writing begins, not when it ends. Polling
    the size until it is stable avoids parsing a half-written file. Returns
    True if the file stabilised, False on timeout or disappearance.
    """
    deadline = time.monotonic() + timeout_s
    last_size = -1
    stable_count = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last_size:
            stable_count += 1
            if stable_count >= polls:
                return True
        else:
            stable_count = 0
            last_size = size
        time.sleep(interval_s)
    logger.warning("File %s did not stabilise within %.0f s", path.name, timeout_s)
    return False


# ──────────────────────────────────────────────────────────────────────
#  2. PURE PROCESSING LOGIC (testable without watchdog)
# ──────────────────────────────────────────────────────────────────────

def process_and_archive(fast_path: Path,
                        processed_dir: Path,
                        failed_dir: Path,
                        results_dir: Path,
                        ) -> bool:
    """Process one file end to end and archive it according to the outcome.

    The file is processed by the orchestrator and its results exported to the
    results directory. On success the source moves to processed_dir; on any
    error it moves to failed_dir and the exception is logged but not raised, so
    the watcher keeps running. Returns True on success, False on failure.
    """
    fast_path = Path(fast_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Processing %s", fast_path.name)
        result = MP.process_case(fast_path)
        Exp.export_result(result, results_dir)
        Viz.plot_pv_experimental(result, results_dir)
        dest = _unique_destination(processed_dir / fast_path.name)
        shutil.move(str(fast_path), str(dest))
        logger.info("Done: %s -> %s  (W_exp=%.1f J)",
                    fast_path.name, dest.parent.name, result.W_experimental_J)
        return True
    except Exception as exc:                       # noqa: BLE001 - we want all
        logger.error("Failed to process %s: %s", fast_path.name, exc)
        try:
            dest = _unique_destination(failed_dir / fast_path.name)
            shutil.move(str(fast_path), str(dest))
        except OSError as move_err:
            logger.error("Could not archive failed file %s: %s",
                         fast_path.name, move_err)
        return False


def _unique_destination(dest: Path) -> Path:
    """Avoid clobbering an existing archived file by adding a numeric suffix."""
    if not dest.exists():
        return dest
    stem, suffix, parent = dest.stem, dest.suffix, dest.parent
    n = 1
    while (candidate := parent / f"{stem}_{n}{suffix}").exists():
        n += 1
    return candidate


# ──────────────────────────────────────────────────────────────────────
#  3. WATCHDOG EVENT ADAPTER
# ──────────────────────────────────────────────────────────────────────

class InboxHandler(FileSystemEventHandler):
    """Trigger processing when a .txt acquisition file appears in the inbox."""

    def __init__(self, processed_dir: Path, failed_dir: Path, results_dir: Path):
        super().__init__()
        self.processed_dir = processed_dir
        self.failed_dir = failed_dir
        self.results_dir = results_dir

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".txt":
            logger.debug("Ignoring non-txt file %s", path.name)
            return
        if not wait_until_stable(path):
            return
        process_and_archive(path, self.processed_dir,
                            self.failed_dir, self.results_dir)


# ──────────────────────────────────────────────────────────────────────
#  4. OBSERVER ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def watch_inbox(root: Path,
               poll_existing: bool = True,
               ) -> None:
    """Start watching root/inbox for new acquisition files (blocking).

    Any .txt files already present when the watcher starts are processed first
    (if poll_existing), then the observer blocks waiting for new arrivals.
    Stop with Ctrl-C.
    """
    root = Path(root)
    inbox     = root / INBOX_DIRNAME
    processed = root / PROCESSED_DIRNAME
    failed    = root / FAILED_DIRNAME
    results   = root / RESULTS_DIRNAME
    inbox.mkdir(parents=True, exist_ok=True)

    if poll_existing:
        for existing in sorted(inbox.glob("*.txt")):
            process_and_archive(existing, processed, failed, results)

    handler = InboxHandler(processed, failed, results)
    observer = _WatchdogObserver()
    observer.schedule(handler, str(inbox), recursive=False)
    observer.start()
    logger.info("Watching %s for new .txt files (Ctrl-C to stop)", inbox)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping observer")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    watch_inbox(Path(__file__).resolve().parent.parent)
