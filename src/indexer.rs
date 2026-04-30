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
}

impl Indexer {
    /// Create a new indexer backed by the given engine and store.
    pub fn new(engine: Arc<EmbeddingEngine>, store: Arc<VectorStore>) -> Self {
        Self { engine, store }
    }

    /// Process a single filesystem event through the full pipeline.
    ///
    /// - **Upsert**: read file → embed content → delete old chunks → insert new chunk.
    /// - **Remove**: delete all chunks for the given path.
    pub async fn process_event(&self, event: FileEvent) -> Result<(), Box<dyn std::error::Error>> {
        match event {
            FileEvent::Upsert(path) => {
                // Small delay to let the OS finish writing the file.
                tokio::time::sleep(Duration::from_millis(100)).await;

                let content = std::fs::read_to_string(&path)?;
                if content.trim().is_empty() {
                    return Ok(());
                }

                println!("[indexer] Embedding: {:?}", path);
                let vector = self.engine.embed(&content).await?;

                let chunk = DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    vector,
                    file_path: path.to_string_lossy().to_string(),
                    content_snippet: content,
                };

                // Replace any previous version of this file.
                self.store.delete_by_path(&chunk.file_path).await?;
                self.store.insert_chunks(&[chunk]).await?;
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
