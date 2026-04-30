use std::sync::Arc;
use std::time::Duration;

use crate::embedding::EmbeddingEngine;
use crate::harvester::FileEvent;
use crate::vectorstore::{DocumentChunk, VectorStore};

/// Owns the `(path, content) → embed → chunk → upsert` indexing pipeline.
///
/// Wraps shared references to the embedding engine and vector store so that
/// callers only need to forward a [`FileEvent`] to get a file fully indexed.
pub struct Indexer {
    engine: Arc<EmbeddingEngine>,
    store: Arc<VectorStore>,
    max_chunk_chars: usize,
}

impl Indexer {
    /// Create a new indexer backed by the given engine and store.
    pub fn new(engine: Arc<EmbeddingEngine>, store: Arc<VectorStore>, max_chunk_chars: usize) -> Self {
        Self { engine, store, max_chunk_chars }
    }

    /// Process a single filesystem event through the full pipeline.
    ///
    /// - **Upsert**: chunk file → embed each chunk → delete old chunks → insert new chunks.
    /// - **Remove**: delete all chunks for the given path.
    pub async fn process_event(&self, event: FileEvent) -> Result<(), Box<dyn std::error::Error>> {
        match event {
            FileEvent::Upsert(path) => {
                // Small delay to let the OS finish writing the file.
                tokio::time::sleep(Duration::from_millis(100)).await;

                let chunks = crate::chunker::chunk_file(&path, self.max_chunk_chars)?;
                if chunks.is_empty() {
                    return Ok(());
                }

                println!("[indexer] Embedding {} chunks for: {:?}", chunks.len(), path);
                
                let file_path_str = path.to_string_lossy().to_string();
                
                // Replace any previous version of this file.
                self.store.delete_by_path(&file_path_str).await?;

                let mut doc_chunks = Vec::with_capacity(chunks.len());
                for chunk in chunks {
                    let vector = self.engine.embed(&chunk.text).await?;
                    doc_chunks.push(DocumentChunk {
                        id: uuid::Uuid::new_v4().to_string(),
                        vector,
                        file_path: file_path_str.clone(),
                        content_snippet: chunk.text,
                    });
                }

                self.store.insert_chunks(&doc_chunks).await?;
            }
            FileEvent::Remove(path) => {
                println!("[indexer] Removing: {:?}", path);
                self.store
                    .delete_by_path(&path.to_string_lossy())
                    .await?;
            }
        }
        Ok(())
    }
}
