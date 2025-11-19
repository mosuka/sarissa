use std::collections::HashMap;
use std::io::{Read, Write};

use crate::error::{PlatypusError, Result};

/// Write a UTF-8 string prefixed by its length as u32 little-endian.
pub fn write_string<W: Write>(output: &mut W, value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    output.write_all(&(bytes.len() as u32).to_le_bytes())?;
    output.write_all(bytes)?;
    Ok(())
}

/// Read a length-prefixed UTF-8 string that was written with [`write_string`].
pub fn read_string<R: Read>(input: &mut R) -> Result<String> {
    let mut len_buf = [0u8; 4];
    input.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    input.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| {
        PlatypusError::InvalidOperation(format!("Invalid UTF-8 sequence in vector metadata: {e}"))
    })
}

/// Write metadata hashmap as a length-prefixed list of key-value pairs.
pub fn write_metadata<W: Write>(output: &mut W, metadata: &HashMap<String, String>) -> Result<()> {
    output.write_all(&(metadata.len() as u32).to_le_bytes())?;
    for (key, value) in metadata {
        write_string(output, key)?;
        write_string(output, value)?;
    }
    Ok(())
}

/// Read metadata hashmap written with [`write_metadata`].
pub fn read_metadata<R: Read>(input: &mut R) -> Result<HashMap<String, String>> {
    let mut count_buf = [0u8; 4];
    input.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut metadata = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = read_string(input)?;
        let value = read_string(input)?;
        metadata.insert(key, value);
    }

    Ok(metadata)
}
