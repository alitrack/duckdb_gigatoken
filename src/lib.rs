use arrow::{
    array::{Int32Array, RecordBatch},
    datatypes::{DataType, Field, Schema},
};
use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    vtab::{arrow::record_batch_to_duckdb_data_chunk, BindInfo, InitInfo, TableFunctionInfo, VTab},
    Connection, Result,
};
use gigatoken_rs::{load_tokenizer::hf::load_hf_bpe, Tokenizer};
use std::{
    collections::HashMap,
    error::Error,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};
use std::sync::atomic::AtomicBool;

// ── Tokenizer registry ──
// Lazy-loaded tokenizers, shared across calls via fork().
static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Tokenizer>>>> = OnceLock::new();

fn get_tokenizer(model_path: &str) -> std::result::Result<Tokenizer, Box<dyn Error>> {
    let map = REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().map_err(|e| format!("lock: {e}"))?;
    if !map.contains_key(model_path) {
        let t = load_hf_bpe(Path::new(model_path))
            .map_err(|e| format!("load tokenizer '{model_path}': {e}"))?;
        map.insert(model_path.to_string(), Arc::new(t));
    }
    let t = map.get(model_path).unwrap();
    Ok(t.fork())
}

// ── gigatoken_encode(model_path, text) → (idx, token_id) ──

#[repr(C)]
pub struct EncodeInitData {
    done: AtomicBool,
}

#[repr(C)]
pub struct EncodeBindData {
    results: Vec<(i32, i32)>,
    schema: Arc<Schema>,
}

pub struct GigatokenEncodeVTab;

impl VTab for GigatokenEncodeVTab {
    type BindData = EncodeBindData;
    type InitData = EncodeInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 2 {
            return Err("gigatoken_encode: expected 2 params (model_path, text)".into());
        }
        let model_path = bind.get_parameter(0).to_string();
        let text = bind.get_parameter(1).to_string();

        let mut tokenizer = get_tokenizer(&model_path)?;
        let mut ids: Vec<u32> = Vec::new();
        tokenizer.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);

        let results: Vec<(i32, i32)> = ids
            .into_iter()
            .enumerate()
            .map(|(i, id)| (i as i32, id as i32))
            .collect();

        let schema = Arc::new(Schema::new(vec![
            Field::new("idx", DataType::Int32, false),
            Field::new("token_id", DataType::Int32, false),
        ]));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("token_id", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(EncodeBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(EncodeInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);

        let bind = func.get_bind_data();
        let n = bind.results.len().min(2048);
        if n == 0 {
            output.set_len(0);
            return Ok(());
        }

        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let tokens: Vec<i32> = bind.results.iter().take(n).map(|r| r.1).collect();

        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(indices)),
                Arc::new(Int32Array::from(tokens)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── gigatoken_encode_batch(model_path, texts) → (doc, idx, token_id) ──

#[repr(C)]
pub struct EncodeBatchBindData {
    results: Vec<(i32, i32, i32)>, // (doc, idx, token_id)
    schema: Arc<Schema>,
}

pub struct GigatokenEncodeBatchVTab;

impl VTab for GigatokenEncodeBatchVTab {
    type BindData = EncodeBatchBindData;
    type InitData = EncodeInitData;

    fn bind(bind: &BindInfo) -> std::result::Result<Self::BindData, Box<dyn Error>> {
        let pc = bind.get_parameter_count();
        if pc < 2 {
            return Err("gigatoken_encode_batch: expected 2 params (model_path, texts[])".into());
        }
        let model_path = bind.get_parameter(0).to_string();
        let texts_str = bind.get_parameter(1).to_string();

        // Parse list of texts: ["text1", "text2", ...]
        let texts = parse_string_list(&texts_str)?;

        let mut tokenizer = get_tokenizer(&model_path)?;
        let mut results = Vec::new();

        for (doc_idx, text) in texts.iter().enumerate() {
            let mut ids: Vec<u32> = Vec::new();
            tokenizer.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
            for (idx, id) in ids.into_iter().enumerate() {
                results.push((doc_idx as i32, idx as i32, id as i32));
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("doc", DataType::Int32, false),
            Field::new("idx", DataType::Int32, false),
            Field::new("token_id", DataType::Int32, false),
        ]));
        bind.add_result_column("doc", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("idx", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("token_id", LogicalTypeHandle::from(LogicalTypeId::Integer));
        Ok(EncodeBatchBindData { results, schema })
    }

    fn init(_: &InitInfo) -> std::result::Result<Self::InitData, Box<dyn Error>> {
        Ok(EncodeInitData { done: false.into() })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> std::result::Result<(), Box<dyn Error>> {
        let init = func.get_init_data();
        if init.done.load(std::sync::atomic::Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }
        init.done.store(true, std::sync::atomic::Ordering::Relaxed);

        let bind = func.get_bind_data();
        let n = bind.results.len().min(2048);
        if n == 0 {
            output.set_len(0);
            return Ok(());
        }

        let docs: Vec<i32> = bind.results.iter().take(n).map(|r| r.0).collect();
        let indices: Vec<i32> = bind.results.iter().take(n).map(|r| r.1).collect();
        let tokens: Vec<i32> = bind.results.iter().take(n).map(|r| r.2).collect();

        let batch = RecordBatch::try_new(
            bind.schema.clone(),
            vec![
                Arc::new(Int32Array::from(docs)),
                Arc::new(Int32Array::from(indices)),
                Arc::new(Int32Array::from(tokens)),
            ],
        )?;
        record_batch_to_duckdb_data_chunk(&batch, output)?;
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
}

// ── Helpers ──

fn parse_string_list(raw: &str) -> std::result::Result<Vec<String>, Box<dyn Error>> {
    let raw = raw.trim();
    if !raw.starts_with('[') || !raw.ends_with(']') {
        return Err("expected JSON array of strings [\"a\", \"b\", ...]".into());
    }
    let inner = &raw[1..raw.len() - 1];
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    let mut result = Vec::new();
    let mut chars = inner.chars().peekable();
    loop {
        // Skip whitespace and commas
        while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ',') {
            chars.next();
        }
        if chars.peek().is_none() {
            break;
        }
        if chars.peek() == Some(&'"') {
            chars.next(); // opening quote
            let mut s = String::new();
            loop {
                match chars.next() {
                    None => return Err("unterminated string".into()),
                    Some('"') => break,
                    Some('\\') => {
                        match chars.next() {
                            Some(c) => s.push(c),
                            None => return Err("unterminated escape".into()),
                        }
                    }
                    Some(c) => s.push(c),
                }
            }
            result.push(s);
        } else {
            return Err(format!("unexpected char: {:?}", chars.peek()).into());
        }
    }
    Ok(result)
}

// ── Entrypoint ──

#[duckdb_entrypoint_c_api(ext_name = "gigatoken")]
pub unsafe fn extension_entrypoint(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<GigatokenEncodeVTab>("gigatoken_encode")?;
    con.register_table_function::<GigatokenEncodeBatchVTab>("gigatoken_encode_batch")?;
    Ok(())
}
