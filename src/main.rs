#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tauri::{GlobalShortcutManager, Manager};

use embi_search::config::AppConfig;
use embi_search::embedding::EmbeddingEngine;
use embi_search::harvester::{scan_directory, spawn_watcher, FileEvent};
use embi_search::indexer::Indexer;
use embi_search::server::LlamaServer;
use embi_search::vectorstore::VectorStore;

// --- App State ---
struct AppState {
    engine: Arc<EmbeddingEngine>,
    store: Arc<VectorStore>,
}

// --- Tauri Commands ---
#[tauri::command]
async fn search_files(
    query: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<embi_search::vectorstore::SearchResult>, String> {
    let vector = state.engine.embed(&query).await
        .map_err(|e| format!("Failed to embed query: {}", e))?;

    let results = state.store.search(&vector, 5).await
        .map_err(|e| format!("Failed to search vector store: {}", e))?;

    Ok(results)
}

#[tauri::command]
fn open_file(path: String) -> Result<(), String> {
    open::that(&path).map_err(|e| format!("Failed to open file: {}", e))?;
    Ok(())
}

// --- Main Application ---
fn main() {
    let config = AppConfig::load();

    tauri::Builder::default()
        .setup(move |app| {
            // 1. Boot the embedding server
            let server = LlamaServer::spawn(&config.llama_dir, &config.model_path, config.server_port)
                .expect("Failed to spawn llama-server");
            server.wait_until_ready(Duration::from_secs(60))
                .expect("Server failed to become ready");

            // 2. Initialize Core Services
            let (engine, store) = tauri::async_runtime::block_on(async {
                let store = VectorStore::connect(&config.db_path, config.vector_dim).await
                    .expect("Failed to connect to LanceDB");

                let engine = EmbeddingEngine::new(server.port(), &config.model_name);

                (Arc::new(engine), Arc::new(store))
            });

            app.manage(AppState {
                engine: Arc::clone(&engine),
                store: Arc::clone(&store),
            });

            app.manage(server);

            // 3. Indexing pipeline — runs entirely inside the Tokio runtime
            let indexer = Arc::new(Indexer::new(Arc::clone(&engine), Arc::clone(&store)));
            let watch_dirs = config.watch_dirs.clone();

            tauri::async_runtime::spawn(async move {
                let (tx, mut rx) = mpsc::channel::<FileEvent>(100);

                // Spawn a watcher + initial scan for EACH configured directory
                let mut watchers = Vec::new();
                for dir in &watch_dirs {
                    std::fs::create_dir_all(dir).unwrap();
                    watchers.push(
                        spawn_watcher(dir.clone(), tx.clone()).expect("Failed to spawn watcher")
                    );
                    let scan_tx = tx.clone();
                    let scan_dir = dir.clone();
                    tokio::spawn(async move {
                        scan_directory(&scan_dir, scan_tx).await;
                    });
                }
                drop(tx); // Drop the original sender; clones in watchers keep the channel open

                while let Some(event) = rx.recv().await {
                    if let Err(e) = indexer.process_event(event).await {
                        eprintln!("[indexer] {}", e);
                    }
                }
            });

            // 4. Window management
            let main_window = app.get_window("main").unwrap();

            // Hide window when it loses focus
            let window_clone = main_window.clone();
            main_window.on_window_event(move |event| {
                if let tauri::WindowEvent::Focused(false) = event {
                    window_clone.hide().unwrap();
                }
            });

            // Register Alt + Space
            let mut shortcut_manager = app.global_shortcut_manager();
            shortcut_manager.register("Alt+Space", move || {
                if main_window.is_visible().unwrap() {
                    main_window.hide().unwrap();
                } else {
                    main_window.show().unwrap();
                    main_window.set_focus().unwrap();
                }
            }).expect("Failed to register global shortcut");

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![search_files, open_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}