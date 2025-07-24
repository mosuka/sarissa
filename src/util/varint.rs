//! Variable-length integer encoding utilities.
//!
//! This module provides efficient variable-length integer encoding and decoding,
//! similar to what's used in protocol buffers and other binary formats.

use crate::error::{Result, SarissaError};
use byteorder::ReadBytesExt;
use std::io::{Read, Write};

/// Encode a u32 value using variable-length encoding.
///
/// Uses 7 bits per byte with a continuation bit, allowing efficient
/// encoding of small numbers.
pub fn encode_u32(value: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut val = value;

    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;

        if val != 0 {
            byte |= 0x80; // Set continuation bit
        }

        bytes.push(byte);

        if val == 0 {
            break;
        }
    }

    bytes
}

/// Decode a u32 value from variable-length encoding.
pub fn decode_u32(bytes: &[u8]) -> Result<(u32, usize)> {
    let mut result = 0u32;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in bytes {
        bytes_read += 1;

        if shift >= 32 {
            return Err(SarissaError::other("VarInt overflow"));
        }

        result |= ((byte & 0x7F) as u32) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }

        shift += 7;
    }

    Err(SarissaError::other("Incomplete VarInt"))
}

/// Encode a u64 value using variable-length encoding.
pub fn encode_u64(value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut val = value;

    loop {
        let mut byte = (val & 0x7F) as u8;
        val >>= 7;

        if val != 0 {
            byte |= 0x80; // Set continuation bit
        }

        bytes.push(byte);

        if val == 0 {
            break;
        }
    }

    bytes
}

/// Decode a u64 value from variable-length encoding.
pub fn decode_u64(bytes: &[u8]) -> Result<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0;
    let mut bytes_read = 0;

    for &byte in bytes {
        bytes_read += 1;

        if shift >= 64 {
            return Err(SarissaError::other("VarInt overflow"));
        }

        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, bytes_read));
        }

        shift += 7;
    }

    Err(SarissaError::other("Incomplete VarInt"))
}

/// Write a variable-length encoded u32 to a writer.
pub fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<usize> {
    let bytes = encode_u32(value);
    writer.write_all(&bytes)?;
    Ok(bytes.len())
}

/// Read a variable-length encoded u32 from a reader.
pub fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut result = 0u32;
    let mut shift = 0;

    loop {
        let byte = reader.read_u8()?;

        if shift >= 32 {
            return Err(SarissaError::other("VarInt overflow"));
        }

        result |= ((byte & 0x7F) as u32) << shift;

        if (byte & 0x80) == 0 {
            return Ok(result);
        }

        shift += 7;
    }
}

/// Write a variable-length encoded u64 to a writer.
pub fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<usize> {
    let bytes = encode_u64(value);
    writer.write_all(&bytes)?;
    Ok(bytes.len())
}

/// Read a variable-length encoded u64 from a reader.
pub fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;

    loop {
        let byte = reader.read_u8()?;

        if shift >= 64 {
            return Err(SarissaError::other("VarInt overflow"));
        }

        result |= ((byte & 0x7F) as u64) << shift;

        if (byte & 0x80) == 0 {
            return Ok(result);
        }

        shift += 7;
    }
}

/// A trait for types that can be encoded as variable-length integers.
pub trait VarInt: Sized {
    /// Encode this value as a variable-length integer.
    fn encode_varint(&self) -> Vec<u8>;

    /// Decode a variable-length integer from bytes.
    fn decode_varint(bytes: &[u8]) -> Result<(Self, usize)>;

    /// Write this value as a variable-length integer to a writer.
    fn write_varint<W: Write>(&self, writer: &mut W) -> Result<usize>;

    /// Read a variable-length integer from a reader.
    fn read_varint<R: Read>(reader: &mut R) -> Result<Self>;
}

impl VarInt for u32 {
    fn encode_varint(&self) -> Vec<u8> {
        encode_u32(*self)
    }

    fn decode_varint(bytes: &[u8]) -> Result<(Self, usize)> {
        decode_u32(bytes)
    }

    fn write_varint<W: Write>(&self, writer: &mut W) -> Result<usize> {
        write_u32(writer, *self)
    }

    fn read_varint<R: Read>(reader: &mut R) -> Result<Self> {
        read_u32(reader)
    }
}

impl VarInt for u64 {
    fn encode_varint(&self) -> Vec<u8> {
        encode_u64(*self)
    }

    fn decode_varint(bytes: &[u8]) -> Result<(Self, usize)> {
        decode_u64(bytes)
    }

    fn write_varint<W: Write>(&self, writer: &mut W) -> Result<usize> {
        write_u64(writer, *self)
    }

    fn read_varint<R: Read>(reader: &mut R) -> Result<Self> {
        read_u64(reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_encode_decode_u32() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, u32::MAX];

        for &value in &test_values {
            let encoded = encode_u32(value);
            let (decoded, bytes_read) = decode_u32(&encoded).unwrap();

            assert_eq!(value, decoded);
            assert_eq!(encoded.len(), bytes_read);
        }
    }

    #[test]
    fn test_encode_decode_u64() {
        let test_values = [0, 1, 127, 128, 255, 256, 16383, 16384, u64::MAX];

        for &value in &test_values {
            let encoded = encode_u64(value);
            let (decoded, bytes_read) = decode_u64(&encoded).unwrap();

            assert_eq!(value, decoded);
            assert_eq!(encoded.len(), bytes_read);
        }
    }

    #[test]
    fn test_varint_trait_u32() {
        let value = 12345u32;
        let encoded = value.encode_varint();
        let (decoded, _) = u32::decode_varint(&encoded).unwrap();

        assert_eq!(value, decoded);
    }

    #[test]
    fn test_varint_trait_u64() {
        let value = 123456789012345u64;
        let encoded = value.encode_varint();
        let (decoded, _) = u64::decode_varint(&encoded).unwrap();

        assert_eq!(value, decoded);
    }

    #[test]
    fn test_write_read_u32() {
        let mut buffer = Vec::new();
        let value = 12345u32;

        let bytes_written = write_u32(&mut buffer, value).unwrap();
        assert_eq!(bytes_written, buffer.len());

        let mut cursor = Cursor::new(buffer);
        let decoded = read_u32(&mut cursor).unwrap();

        assert_eq!(value, decoded);
    }

    #[test]
    fn test_write_read_u64() {
        let mut buffer = Vec::new();
        let value = 123456789012345u64;

        let bytes_written = write_u64(&mut buffer, value).unwrap();
        assert_eq!(bytes_written, buffer.len());

        let mut cursor = Cursor::new(buffer);
        let decoded = read_u64(&mut cursor).unwrap();

        assert_eq!(value, decoded);
    }

    #[test]
    fn test_varint_trait_write_read() {
        let mut buffer = Vec::new();
        let value = 98765u32;

        let bytes_written = value.write_varint(&mut buffer).unwrap();
        assert_eq!(bytes_written, buffer.len());

        let mut cursor = Cursor::new(buffer);
        let decoded = u32::read_varint(&mut cursor).unwrap();

        assert_eq!(value, decoded);
    }

    #[test]
    fn test_encoding_efficiency() {
        // Small values should use fewer bytes
        assert_eq!(encode_u32(0).len(), 1);
        assert_eq!(encode_u32(127).len(), 1);
        assert_eq!(encode_u32(128).len(), 2);
        assert_eq!(encode_u32(16383).len(), 2);
        assert_eq!(encode_u32(16384).len(), 3);

        // Large values should use more bytes
        assert!(encode_u32(u32::MAX).len() <= 5);
        assert!(encode_u64(u64::MAX).len() <= 10);
    }

    #[test]
    fn test_incomplete_varint() {
        // Test with incomplete data (continuation bit set but no more bytes)
        let incomplete = vec![0x80]; // Continuation bit set but no more data
        assert!(decode_u32(&incomplete).is_err());
        assert!(decode_u64(&incomplete).is_err());
    }

    #[test]
    fn test_overflow() {
        // Test with data that would overflow
        let overflow_data = vec![0xFF; 10]; // Too many bytes for u32
        let result = decode_u32(&overflow_data);
        assert!(result.is_err());
    }
}
