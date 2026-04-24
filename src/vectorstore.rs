use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::Table as LanceTable;
use serde::{Deserialize, Serialize};

/// Dimensionality of the embedding vectors (Nomic Embed Text v1.5).
pub const VECTOR_DIM: i32 = 768;

/// A document chunk ready to be inserted into the vector store.
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub vector: Vec<f32>,
    pub file_path: String,
    pub content_snippet: String,
}

/// A search result returned from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub file_path: String,
    pub content_snippet: String,
    pub distance: f32,
}

/// Persistent vector store backed by LanceDB.
///
/// Stores document chunk embeddings on disk and provides sub-millisecond
/// nearest-neighbor retrieval via the Lance columnar format.
pub struct VectorStore {
    db: Connection,
    table_name: String,
}

impl VectorStore {
    /// Connect to (or create) a LanceDB database at `db_path` and ensure the
    /// `documents` table exists with the correct schema.
    pub async fn connect(db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let db = lancedb::connect(db_path).execute().await?;
        let table_name = "documents".to_string();

        // Check if the table already exists.
        let tables = db.table_names().execute().await?;
        if !tables.contains(&table_name) {
            // Create an empty table with our schema.
            let schema = Self::schema();
            db.create_empty_table(&table_name, schema)
                .execute()
                .await?;
            println!("[vectorstore] Created new '{}' table", table_name);
        } else {
            println!("[vectorstore] Opened existing '{}' table", table_name);
        }

        Ok(Self { db, table_name })
    }

    /// Return the Arrow schema for the documents table.
    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    VECTOR_DIM,
                ),
                true,
            ),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("content_snippet", DataType::Utf8, false),
        ]))
    }

    /// Open the table handle.
    async fn table(&self) -> Result<LanceTable, Box<dyn std::error::Error>> {
        Ok(self.db.open_table(&self.table_name).execute().await?)
    }

    /// Insert one or more document chunks into the store.
    pub async fn insert_chunks(
        &self,
        chunks: &[DocumentChunk],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if chunks.is_empty() {
            return Ok(());
        }

        let schema = Self::schema();
        let len = chunks.len();

        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let paths: Vec<&str> = chunks.iter().map(|c| c.file_path.as_str()).collect();
        let snippets: Vec<&str> = chunks.iter().map(|c| c.content_snippet.as_str()).collect();

        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            chunks.iter().map(|c| Some(c.vector.iter().map(|v| Some(*v)).collect::<Vec<_>>())),
            VECTOR_DIM,
        );

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(vectors),
                Arc::new(StringArray::from(paths)),
                Arc::new(StringArray::from(snippets)),
            ],
        )?;

        let table = self.table().await?;
        let reader = RecordBatchIterator::new(vec![Ok(batch)], Self::schema());
        table.add(reader).execute().await?;

        println!("[vectorstore] Inserted {} chunk(s)", len);
        Ok(())
    }

    /// Search for the nearest neighbors of `query_vector`.
    ///
    /// Returns up to `limit` results sorted by L2 distance (ascending).
    pub async fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let table = self.table().await?;

        let batches: Vec<RecordBatch> = table
            .query()
            .nearest_to(query_vector)?
            .limit(limit)
            .execute()
            .await?
            .try_collect()
            .await?;

        let mut results = Vec::new();

        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let path_col = batch
                .column_by_name("file_path")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let snippet_col = batch
                .column_by_name("content_snippet")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let dist_col = batch
                .column_by_name("_distance")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                results.push(SearchResult {
                    id: id_col.value(i).to_string(),
                    file_path: path_col.value(i).to_string(),
                    content_snippet: snippet_col.value(i).to_string(),
                    distance: dist_col.value(i),
                });
            }
        }

        Ok(results)
    }

    /// Delete all chunks associated with the given file path.
    pub async fn delete_by_path(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let table = self.table().await?;
        let predicate = format!("file_path = '{}'", file_path.replace('\'', "''"));
        table.delete(&predicate).await?;
        println!("[vectorstore] Deleted chunks for: {}", file_path);
        Ok(())
    }

    /// Create an automatic vector index for faster retrieval at scale.
    ///
    /// This is a no-op on very small tables but becomes critical when the
    /// document count reaches thousands. LanceDB will pick IVF-PQ internally.
    pub async fn create_index(&self) -> Result<(), Box<dyn std::error::Error>> {
        let table = self.table().await?;
        table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await?;
        println!("[vectorstore] Created vector index");
        Ok(())
    }

    /// Return the number of rows in the table.
    pub async fn count_rows(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let table = self.table().await?;
        Ok(table.count_rows(None).await?)
    }
}
