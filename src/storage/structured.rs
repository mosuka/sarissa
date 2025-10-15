//! Structured file I/O for binary data serialization.
//!
//! This module provides efficient binary serialization for search index data structures,
//! similar to Whoosh's structfile.py but optimized for Rust and modern hardware.

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::error::{Result, SageError};
use crate::storage::{StorageInput, StorageOutput};
use crate::util::varint::{decode_u64, encode_u64};

/// A structured file writer for binary data.
pub struct StructWriter<W: StorageOutput> {
    writer: W,
    checksum: u32,
    position: u64,
}

impl<W: StorageOutput> StructWriter<W> {
    /// Create a new structured file writer.
    pub fn new(writer: W) -> Self {
        StructWriter {
            writer,
            checksum: 0,
            position: 0,
        }
    }

    /// Write a u8 value.
    pub fn write_u8(&mut self, value: u8) -> Result<()> {
        self.writer.write_u8(value)?;
        self.update_checksum(&[value]);
        self.position += 1;
        Ok(())
    }

    /// Write a u16 value (little-endian).
    pub fn write_u16(&mut self, value: u16) -> Result<()> {
        self.writer.write_u16::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 2;
        Ok(())
    }

    /// Write a u32 value (little-endian).
    pub fn write_u32(&mut self, value: u32) -> Result<()> {
        self.writer.write_u32::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(())
    }

    /// Write a u64 value (little-endian).
    pub fn write_u64(&mut self, value: u64) -> Result<()> {
        self.writer.write_u64::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(())
    }

    /// Write a variable-length integer.
    pub fn write_varint(&mut self, value: u64) -> Result<()> {
        let encoded = encode_u64(value);
        self.writer.write_all(&encoded)?;
        self.update_checksum(&encoded);
        self.position += encoded.len() as u64;
        Ok(())
    }

    /// Write a f32 value (little-endian).
    pub fn write_f32(&mut self, value: f32) -> Result<()> {
        self.writer.write_f32::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(())
    }

    /// Write a f64 value (little-endian).
    pub fn write_f64(&mut self, value: f64) -> Result<()> {
        self.writer.write_f64::<LittleEndian>(value)?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(())
    }

    /// Write a string with length prefix.
    pub fn write_string(&mut self, value: &str) -> Result<()> {
        let bytes = value.as_bytes();
        self.write_varint(bytes.len() as u64)?;
        self.writer.write_all(bytes)?;
        self.update_checksum(bytes);
        self.position += bytes.len() as u64;
        Ok(())
    }

    /// Write raw bytes with length prefix.
    pub fn write_bytes(&mut self, value: &[u8]) -> Result<()> {
        self.write_varint(value.len() as u64)?;
        self.writer.write_all(value)?;
        self.update_checksum(value);
        self.position += value.len() as u64;
        Ok(())
    }

    /// Write raw bytes without length prefix.
    pub fn write_raw(&mut self, value: &[u8]) -> Result<()> {
        self.writer.write_all(value)?;
        self.update_checksum(value);
        self.position += value.len() as u64;
        Ok(())
    }

    /// Write a compressed integer array using delta encoding.
    pub fn write_delta_compressed_u32s(&mut self, values: &[u32]) -> Result<()> {
        if values.is_empty() {
            return self.write_varint(0);
        }

        self.write_varint(values.len() as u64)?;

        let mut previous = 0u32;
        for &value in values {
            let delta = value.wrapping_sub(previous);
            self.write_varint(delta as u64)?;
            previous = value;
        }

        Ok(())
    }

    /// Write a hash map with string keys and u64 values.
    pub fn write_string_u64_map(&mut self, map: &HashMap<String, u64>) -> Result<()> {
        self.write_varint(map.len() as u64)?;

        for (key, value) in map {
            self.write_string(key)?;
            self.write_u64(*value)?;
        }

        Ok(())
    }

    /// Get current file position.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get current checksum.
    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    /// Update checksum with new data.
    fn update_checksum(&mut self, data: &[u8]) {
        self.checksum = crc32fast::hash(data);
    }

    /// Flush and close the writer.
    pub fn close(mut self) -> Result<()> {
        // Write final checksum
        self.writer.write_u32::<LittleEndian>(self.checksum)?;
        self.writer.flush_and_sync()?;
        self.writer.close()?;
        Ok(())
    }
}

/// A structured file reader for binary data.
pub struct StructReader<R: StorageInput> {
    reader: R,
    checksum: u32,
    position: u64,
    file_size: u64,
}

impl<R: StorageInput> StructReader<R> {
    /// Create a new structured file reader.
    pub fn new(reader: R) -> Result<Self> {
        let file_size = reader.size()?;
        Ok(StructReader {
            reader,
            checksum: 0,
            position: 0,
            file_size,
        })
    }

    /// Read a u8 value.
    pub fn read_u8(&mut self) -> Result<u8> {
        let value = self.reader.read_u8()?;
        self.update_checksum(&[value]);
        self.position += 1;
        Ok(value)
    }

    /// Read a u16 value (little-endian).
    pub fn read_u16(&mut self) -> Result<u16> {
        let value = self.reader.read_u16::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 2;
        Ok(value)
    }

    /// Read a u32 value (little-endian).
    pub fn read_u32(&mut self) -> Result<u32> {
        let value = self.reader.read_u32::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(value)
    }

    /// Read a u64 value (little-endian).
    pub fn read_u64(&mut self) -> Result<u64> {
        let value = self.reader.read_u64::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(value)
    }

    /// Read a variable-length integer.
    pub fn read_varint(&mut self) -> Result<u64> {
        let mut bytes = Vec::new();
        loop {
            let byte = self.reader.read_u8()?;
            bytes.push(byte);
            if byte & 0x80 == 0 {
                break;
            }
        }

        let (value, _) = decode_u64(&bytes)?;
        self.update_checksum(&bytes);
        self.position += bytes.len() as u64;
        Ok(value)
    }

    /// Read a f32 value (little-endian).
    pub fn read_f32(&mut self) -> Result<f32> {
        let value = self.reader.read_f32::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 4;
        Ok(value)
    }

    /// Read a f64 value (little-endian).
    pub fn read_f64(&mut self) -> Result<f64> {
        let value = self.reader.read_f64::<LittleEndian>()?;
        self.update_checksum(&value.to_le_bytes());
        self.position += 8;
        Ok(value)
    }

    /// Read a string with length prefix.
    pub fn read_string(&mut self) -> Result<String> {
        let length = self.read_varint()? as usize;
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;

        String::from_utf8(bytes).map_err(|e| SageError::storage(format!("Invalid UTF-8: {e}")))
    }

    /// Read bytes with length prefix.
    pub fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let length = self.read_varint()? as usize;
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;
        Ok(bytes)
    }

    /// Read exact number of raw bytes.
    pub fn read_raw(&mut self, length: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; length];
        self.reader.read_exact(&mut bytes)?;
        self.update_checksum(&bytes);
        self.position += length as u64;
        Ok(bytes)
    }

    /// Read a delta-compressed integer array.
    pub fn read_delta_compressed_u32s(&mut self) -> Result<Vec<u32>> {
        let length = self.read_varint()? as usize;
        if length == 0 {
            return Ok(Vec::new());
        }

        let mut values = Vec::with_capacity(length);
        let mut previous = 0u32;

        for _ in 0..length {
            let delta = self.read_varint()? as u32;
            let value = previous.wrapping_add(delta);
            values.push(value);
            previous = value;
        }

        Ok(values)
    }

    /// Read a hash map with string keys and u64 values.
    pub fn read_string_u64_map(&mut self) -> Result<HashMap<String, u64>> {
        let length = self.read_varint()? as usize;
        let mut map = HashMap::with_capacity(length);

        for _ in 0..length {
            let key = self.read_string()?;
            let value = self.read_u64()?;
            map.insert(key, value);
        }

        Ok(map)
    }

    /// Get current file position.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Get file size.
    pub fn size(&self) -> u64 {
        self.file_size
    }

    /// Check if we're at end of file.
    pub fn is_eof(&self) -> bool {
        self.position >= self.file_size.saturating_sub(4) // Account for checksum
    }

    /// Get current checksum.
    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    /// Update checksum with new data.
    fn update_checksum(&mut self, data: &[u8]) {
        self.checksum = crc32fast::hash(data);
    }

    /// Verify file integrity by checking final checksum.
    pub fn verify_checksum(&mut self) -> Result<bool> {
        if self.position + 4 > self.file_size {
            return Err(SageError::storage("File too short for checksum"));
        }

        // Read the stored checksum from the end of file
        let stored_checksum = self.reader.read_u32::<LittleEndian>()?;
        Ok(stored_checksum == self.checksum)
    }

    /// Close the reader.
    pub fn close(mut self) -> Result<()> {
        self.reader.close()
    }
}

/// Efficient block-based I/O for posting lists.
pub struct BlockWriter<W: StorageOutput> {
    writer: StructWriter<W>,
    block_size: usize,
    current_block: Vec<u8>,
    blocks_written: u64,
}

impl<W: StorageOutput> BlockWriter<W> {
    /// Create a new block writer.
    pub fn new(writer: W, block_size: usize) -> Self {
        BlockWriter {
            writer: StructWriter::new(writer),
            block_size,
            current_block: Vec::with_capacity(block_size),
            blocks_written: 0,
        }
    }

    /// Write data to the current block.
    pub fn write_to_block(&mut self, data: &[u8]) -> Result<()> {
        if self.current_block.len() + data.len() > self.block_size {
            self.flush_block()?;
        }

        if data.len() > self.block_size {
            // Data is larger than block size, write directly
            self.writer.write_raw(data)?;
        } else {
            self.current_block.extend_from_slice(data);
        }

        Ok(())
    }

    /// Flush the current block to storage.
    pub fn flush_block(&mut self) -> Result<()> {
        if !self.current_block.is_empty() {
            // Write block header: size + block number
            self.writer.write_u32(self.current_block.len() as u32)?;
            self.writer.write_u64(self.blocks_written)?;

            // Write block data
            self.writer.write_raw(&self.current_block)?;

            self.current_block.clear();
            self.blocks_written += 1;
        }
        Ok(())
    }

    /// Get the number of blocks written.
    pub fn blocks_written(&self) -> u64 {
        self.blocks_written
    }

    /// Close the writer.
    pub fn close(mut self) -> Result<()> {
        self.flush_block()?;
        self.writer.close()
    }
}

/// Efficient block-based reader for posting lists.
pub struct BlockReader<R: StorageInput> {
    reader: StructReader<R>,
    block_cache: Vec<u8>,
    current_block_size: usize,
    current_block_pos: usize,
    blocks_read: u64,
}

impl<R: StorageInput> BlockReader<R> {
    /// Create a new block reader.
    pub fn new(reader: R) -> Result<Self> {
        Ok(BlockReader {
            reader: StructReader::new(reader)?,
            block_cache: Vec::new(),
            current_block_size: 0,
            current_block_pos: 0,
            blocks_read: 0,
        })
    }

    /// Read the next block.
    pub fn read_block(&mut self) -> Result<Option<&[u8]>> {
        if self.reader.is_eof() {
            return Ok(None);
        }

        // Read block header
        let block_size = self.reader.read_u32()? as usize;
        let block_number = self.reader.read_u64()?;

        // Verify block number
        if block_number != self.blocks_read {
            return Err(SageError::storage(format!(
                "Block number mismatch: expected {}, got {}",
                self.blocks_read, block_number
            )));
        }

        // Read block data
        self.block_cache = self.reader.read_raw(block_size)?;
        self.current_block_size = block_size;
        self.current_block_pos = 0;
        self.blocks_read += 1;

        Ok(Some(&self.block_cache))
    }

    /// Read data from the current block.
    pub fn read_from_block(&mut self, length: usize) -> Result<Option<&[u8]>> {
        if self.current_block_pos + length > self.current_block_size {
            return Ok(None);
        }

        let start = self.current_block_pos;
        let end = start + length;
        self.current_block_pos = end;

        Ok(Some(&self.block_cache[start..end]))
    }

    /// Get the number of blocks read.
    pub fn blocks_read(&self) -> u64 {
        self.blocks_read
    }

    /// Close the reader.
    pub fn close(self) -> Result<()> {
        self.reader.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{MemoryStorage, Storage, StorageConfig};
    use std::sync::Arc;

    #[test]
    fn test_struct_writer_reader() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        // Write structured data
        {
            let output = storage.create_output("test.struct").unwrap();
            let mut writer = StructWriter::new(output);

            writer.write_u8(42).unwrap();
            writer.write_u16(1234).unwrap();
            writer.write_u32(5678).unwrap();
            writer.write_u64(9876543210).unwrap();
            writer.write_varint(12345).unwrap();
            writer.write_f32(std::f32::consts::PI).unwrap();
            writer.write_f64(std::f64::consts::E).unwrap();
            writer.write_string("Hello, World!").unwrap();
            writer.write_bytes(b"binary data").unwrap();

            let values = vec![1, 5, 10, 15, 25];
            writer.write_delta_compressed_u32s(&values).unwrap();

            writer.close().unwrap();
        }

        // Read structured data
        {
            let input = storage.open_input("test.struct").unwrap();
            let mut reader = StructReader::new(input).unwrap();

            assert_eq!(reader.read_u8().unwrap(), 42);
            assert_eq!(reader.read_u16().unwrap(), 1234);
            assert_eq!(reader.read_u32().unwrap(), 5678);
            assert_eq!(reader.read_u64().unwrap(), 9876543210);
            assert_eq!(reader.read_varint().unwrap(), 12345);
            assert!((reader.read_f32().unwrap() - std::f32::consts::PI).abs() < 0.0001);
            assert!((reader.read_f64().unwrap() - std::f64::consts::E).abs() < 0.000000001);
            assert_eq!(reader.read_string().unwrap(), "Hello, World!");
            assert_eq!(reader.read_bytes().unwrap(), b"binary data");

            let decoded_values = reader.read_delta_compressed_u32s().unwrap();
            assert_eq!(decoded_values, vec![1, 5, 10, 15, 25]);

            // Verify checksum
            assert!(reader.verify_checksum().unwrap());
        }
    }

    #[test]
    fn test_block_writer_reader() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        // Write blocks
        {
            let output = storage.create_output("test.blocks").unwrap();
            let mut writer = BlockWriter::new(output, 1024);

            writer.write_to_block(b"First block data").unwrap();
            writer.write_to_block(b"More data in first block").unwrap();
            writer.flush_block().unwrap();

            writer.write_to_block(b"Second block data").unwrap();
            writer.close().unwrap();
        }

        // Read blocks
        {
            let input = storage.open_input("test.blocks").unwrap();
            let mut reader = BlockReader::new(input).unwrap();

            // Read first block
            let block1 = reader.read_block().unwrap().unwrap();
            assert!(block1.starts_with(b"First block data"));

            // Read second block
            let block2 = reader.read_block().unwrap().unwrap();
            assert!(block2.starts_with(b"Second block data"));

            // No more blocks
            assert!(reader.read_block().unwrap().is_none());

            reader.close().unwrap();
        }
    }

    #[test]
    fn test_string_u64_map() {
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        let mut original_map = HashMap::new();
        original_map.insert("term1".to_string(), 100);
        original_map.insert("term2".to_string(), 200);
        original_map.insert("term3".to_string(), 300);

        // Write map
        {
            let output = storage.create_output("test.map").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_string_u64_map(&original_map).unwrap();
            writer.close().unwrap();
        }

        // Read map
        {
            let input = storage.open_input("test.map").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let read_map = reader.read_string_u64_map().unwrap();

            assert_eq!(read_map.len(), original_map.len());
            for (key, value) in &original_map {
                assert_eq!(read_map.get(key), Some(value));
            }

            reader.close().unwrap();
        }
    }

    #[test]
    fn test_delta_compression() {
        let values = vec![1000, 1005, 1010, 1020, 1050, 1100];
        let storage = Arc::new(MemoryStorage::new(StorageConfig::default()));

        // Write compressed values
        {
            let output = storage.create_output("test.delta").unwrap();
            let mut writer = StructWriter::new(output);
            writer.write_delta_compressed_u32s(&values).unwrap();
            writer.close().unwrap();
        }

        // Read and verify
        {
            let input = storage.open_input("test.delta").unwrap();
            let mut reader = StructReader::new(input).unwrap();
            let decoded = reader.read_delta_compressed_u32s().unwrap();
            assert_eq!(decoded, values);
            reader.close().unwrap();
        }
    }
}
