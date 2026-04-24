use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::sleep;

use embi_search::embedding::EmbeddingEngine;
use embi_search::harvester::{spawn_watcher, FileEvent};
use embi_search::server::LlamaServer;
use embi_search::vectorstore::{DocumentChunk, VectorStore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llama_dir = Path::new("llama");
    let model_path = Path::new("model/nomic-embed-text-v1.5.Q4_K_M.gguf");
    let port: u16 = 8080;

    // ── Boot the embedding server ───────────────────────────────────────
    let server = LlamaServer::spawn(llama_dir, model_path, port)?;
    server.wait_until_ready(Duration::from_secs(60))?;

    let rt = tokio::runtime::Runtime::new()?;
    
    let _guard = rt.enter();
    // Wrap our core services in Arc so they can be safely shared with background threads
    let engine = Arc::new(EmbeddingEngine::new(server.port()));
    
    let test_db_path = ".lancedb_test";
    if Path::new(test_db_path).exists() {
        std::fs::remove_dir_all(test_db_path)?;
    }
    let store = Arc::new(rt.block_on(VectorStore::connect(test_db_path))?);

    // ── Phase 3: Watcher Pipeline Test ──────────────────────────────────
    println!("\n=== Phase 3: Watcher Pipeline Test ===");

    let test_dir = PathBuf::from("test_watch_zone");
    if test_dir.exists() {
        std::fs::remove_dir_all(&test_dir)?;
    }
    std::fs::create_dir(&test_dir)?;

    // 1. Create the channel
    let (tx, mut rx) = mpsc::channel::<FileEvent>(100);

    // 2. Start the Watcher on the test directory
    let _watcher = spawn_watcher(test_dir.clone(), tx.clone())?;
    println!("  Watcher spawned on {:?}", test_dir);

    // 3. Spawn the background Worker Task
   // 3. Spawn the background Worker Task
    let engine_clone = Arc::clone(&engine);
    let store_clone = Arc::clone(&store);
    
    let worker_handle = rt.spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                FileEvent::Upsert(path) => {
                    // Windows File Lock mitigation
                    sleep(Duration::from_millis(100)).await;
                    
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if content.trim().is_empty() { continue; }
                        
                        println!("  [Worker] Upserting: {:?}", path);
                        
                        // FIX: Use .ok() to convert Result to Option. 
                        // This drops the non-Send Box<dyn Error> immediately!
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
                    println!("  [Worker] Removing: {:?}", path);
                    let _ = store_clone.delete_by_path(&path.to_string_lossy()).await;
                }
            }
        }
    });
    // 4. Simulate User Filesystem Actions
    let file_path = test_dir.join("live_test.txt");
    
    // Create
    std::fs::write(&file_path, "Rust makes file watching safe and easy.")?;
    std::thread::sleep(Duration::from_secs(2)); // Wait for pipeline
    let count = rt.block_on(store.count_rows())?;
    println!("  Rows after creation: {}", count);
    assert!(count >= 1, "Database should have stored the new file");

    // Modify
    std::fs::write(&file_path, "Rust makes file watching highly performant.")?;
    std::thread::sleep(Duration::from_secs(2));

    // Delete
    std::fs::remove_file(&file_path)?;
    std::thread::sleep(Duration::from_secs(2));
    
    let final_count = rt.block_on(store.count_rows())?;
    println!("  Rows after deletion: {}", final_count);
    assert_eq!(final_count, 0, "Database should be empty after deletion");

    drop(_watcher); 
    drop(tx);       


    rt.block_on(worker_handle).ok(); 

    std::fs::remove_dir_all(&test_dir)?;
    std::fs::remove_dir_all(test_db_path)?;
    println!("  ✓ Phase 3 PASSED");

    println!("\n=== All tests passed! ===");
    println!("Server will be shut down.");

    drop(rt); 
    Ok(())
}