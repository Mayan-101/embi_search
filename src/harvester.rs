use ignore::WalkBuilder;
use notify::{EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc;

/// Represents an actionable filesystem event.
#[derive(Debug)]
pub enum FileEvent {
    Upsert(PathBuf),
    Remove(PathBuf),
}

/// Basic filter to only process plaintext and code files.
pub fn is_supported(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext, "txt" | "md" | "rs" | "py" | "csv" | "json" | "toml")
}

/// Recursively walks a directory, skipping hidden files and respecting .gitignore.
/// Sends discovered valid files to the provided channel.
pub async fn scan_directory(root: &Path, tx: mpsc::Sender<FileEvent>) {
    let walker = WalkBuilder::new(root)
        .standard_filters(true)
        .build();

    for result in walker.into_iter().flatten() {
        if result.file_type().map_or(false, |ft| ft.is_file()) {
            let path = result.into_path();
            if is_supported(&path) {
                let _ = tx.send(FileEvent::Upsert(path)).await;
            }
        }
    }
}

/// Spawns an OS-level filesystem watcher.
/// Bridges the synchronous `notify` crate to our async Tokio channel.
pub fn spawn_watcher(
    watch_path: PathBuf,
    tx: mpsc::Sender<FileEvent>,
) -> Result<notify::RecommendedWatcher, Box<dyn std::error::Error>> {
    // notify uses standard sync channels
    let (std_tx, std_rx) = std_mpsc::channel();

    let mut watcher = notify::recommended_watcher(std_tx)?;
    watcher.watch(&watch_path, RecursiveMode::Recursive)?;

    // Spawn a dedicated blocking thread to translate OS events to our async pipeline
    tokio::task::spawn_blocking(move || {
        for res in std_rx {
            if let Ok(event) = res {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                        for path in event.paths {
                            if is_supported(&path) {
                                let _ = tx.blocking_send(FileEvent::Upsert(path));
                            }
                        }
                    }
                    EventKind::Remove(_) => {
                        for path in event.paths {
                            if is_supported(&path) {
                                let _ = tx.blocking_send(FileEvent::Remove(path));
                            }
                        }
                    }
                    _ => {} // Ignore access reads or metadata changes
                }
            }
        }
    });

    Ok(watcher)
}