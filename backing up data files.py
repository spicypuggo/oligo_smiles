import os
import shutil
import schedule
import time
from datetime import datetime
import hashlib

src = r'C:SOURCE PATH HERE'
dst_root = r'C:DESTINATION PATH HERE'

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_all_previous_backups():
    backups = [
        os.path.join(dst_root, d) for d in os.listdir(dst_root)
        if os.path.isdir(os.path.join(dst_root, d)) and d.startswith("backup_")
    ]
    return sorted(backups, key=os.path.getmtime)

def file_exists_in_backups(rel_path, src_hash, backups):
    for backup in backups:
        potential_file = os.path.join(backup, rel_path)
        if os.path.exists(potential_file):
            if compute_file_hash(potential_file) == src_hash:
                return True
    return False

def backup_if_new(src_dir, dst_dir, all_backups):
    files_copied = 0
    for root, _, files in os.walk(src_dir):
        rel_dir = os.path.relpath(root, src_dir)
        dst_subdir = os.path.join(dst_dir, rel_dir)
        os.makedirs(dst_subdir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.join(rel_dir, file)
            file_hash = compute_file_hash(src_file)

            if file_exists_in_backups(rel_path, file_hash, all_backups):
                continue  # file already backed up before

            dst_file = os.path.join(dst_subdir, file)
            shutil.copy2(src_file, dst_file)
            print(f"✔ Copied: {src_file} → {dst_file}")
            files_copied += 1

    print(f"✅ Backup complete: {files_copied} new/changed files copied.")

def incremental_backup_across_all():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_backup = os.path.join(dst_root, f"backup_{timestamp}")
    all_previous_backups = get_all_previous_backups()

    print(f"📦 Starting backup to: {new_backup}")
    backup_if_new(src, new_backup, all_previous_backups)

schedule.every(1).minute.do(incremental_backup_across_all)

while True:
    schedule.run_pending()
    time.sleep(60)
