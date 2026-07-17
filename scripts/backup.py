"""Backup data/ and models/ into a timestamped folder under backups/.

Usage:
    python scripts/backup.py

Meant to be run manually before risky changes (retraining, data updates) or
scheduled via cron/Task Scheduler for periodic backups. Keeps the last
KEEP_LAST backups and prunes older ones automatically.
"""
import os
import shutil
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SOURCES = ["data", "models"]
BACKUPS_DIR = os.path.join(BASE_DIR, "backups")
KEEP_LAST = 5


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUPS_DIR, f"backup_{timestamp}")
    os.makedirs(dest, exist_ok=True)

    for name in SOURCES:
        src = os.path.join(BASE_DIR, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, name))
            print(f"Backed up {name}/ -> {dest}/{name}/")
        else:
            print(f"Skipped {name}/ (not found)")

    _prune_old_backups()
    print(f"\nBackup selesai: {dest}")


def _prune_old_backups():
    if not os.path.isdir(BACKUPS_DIR):
        return
    entries = sorted(
        (e for e in os.listdir(BACKUPS_DIR) if e.startswith("backup_")),
        reverse=True,
    )
    for old in entries[KEEP_LAST:]:
        shutil.rmtree(os.path.join(BACKUPS_DIR, old), ignore_errors=True)
        print(f"Menghapus backup lama: {old}")


if __name__ == "__main__":
    main()
