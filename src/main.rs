#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tauri::{GlobalShortcutManager, Manager};

use embi_search::embedding::EmbeddingEngine;
use embi_search::harvester::{spawn_watcher, FileEvent};
use embi_search::server::LlamaServer;
use embi_search::vectorstore::{DocumentChunk, VectorStore};

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
    tauri::Builder::default()
        .setup(|app| {
            // 1. Boot the embedding server
            let llama_dir = Path::new("llama");
            let model_path = Path::new("model/nomic-embed-text-v1.5.Q4_K_M.gguf");
            
            // We manage the server lifecycle manually here. 
            let server = LlamaServer::spawn(llama_dir, model_path, 8080)
                .expect("Failed to spawn llama-server");
            server.wait_until_ready(Duration::from_secs(60))
                .expect("Server failed to become ready");

            // 2. Initialize Core Services using Tauri's async runtime
            let (engine, store) = tauri::async_runtime::block_on(async {
                let db_path = ".lancedb_data"; // Persistent DB now
                let store = VectorStore::connect(db_path).await
                    .expect("Failed to connect to LanceDB");
                
                let engine = EmbeddingEngine::new(server.port());
                
                (Arc::new(engine), Arc::new(store))
            });

            app.manage(AppState {
                engine: Arc::clone(&engine),
                store: Arc::clone(&store),
            });
            
            app.manage(server);

            // 4. Start the Harvester & Watcher
            // For now, we will just watch a specific test directory. 
            let watch_dir = PathBuf::from("test_watch_zone");
            if !watch_dir.exists() {
                std::fs::create_dir_all(&watch_dir).unwrap();
            }

            let (tx, mut rx) = mpsc::channel::<FileEvent>(100);

            // 5. Spawn the background worker onto Tauri's async runtime
            let engine_clone = Arc::clone(&engine);
            let store_clone = Arc::clone(&store);
            
            tauri::async_runtime::spawn(async move {
                // Keep the watcher alive by holding its reference in this loop's closure
                let _watcher = spawn_watcher(watch_dir, tx).expect("Failed to spawn watcher");
                let _kept_watcher = _watcher; 
                while let Some(event) = rx.recv().await {
                    match event {
                        FileEvent::Upsert(path) => {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            if let Ok(content) = std::fs::read_to_string(&path) {
                                if content.trim().is_empty() { continue; }
                                println!("[Worker] Embedding: {:?}", path);
                                if let Some(vector) = engine_clone.embed(&content).await.ok() {
                                    let chunk = DocumentChunk {
                                        id: uuid::Uuid::new_v4().to_string(),
                                        vector,
                                        file_path: path.to_string_lossy().to_string(),
                                        content_snippet: content,
                                    };
                                    let _ = store_clone.delete_by_path(&chunk.file_path).await;
                                    let _ = store_clone.insert_chunks(&[chunk]).await;
                                }
                            }
                        }
                        FileEvent::Remove(path) => {
                            println!("[Worker] Removing: {:?}", path);
                            let _ = store_clone.delete_by_path(&path.to_string_lossy()).await;
                        }
                    }
                }
            });

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