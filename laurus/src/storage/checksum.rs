//! CRC-32 streaming adapters for integrity-checked file formats (Issue #786).
//!
//! [`CrcWriter`] and [`CrcReader`] are thin pass-through wrappers that maintain
//! a running CRC-32 ([`crc32fast`]) over the bytes written or read, plus a byte
//! counter. They let a format that writes/reads its payload directly (e.g. the
//! HNSW `.hnsw` segment) accumulate a checksum *during* its existing I/O — so a
//! CRC footer can be appended on write and verified on read with no extra pass
//! over the data.
//!
//! Unlike [`crate::storage::structured::StructWriter`], these add no framing:
//! the bytes on disk are exactly the bytes written, and the checksum is the
//! caller's to place (typically as a trailing footer).

use std::io::{Read, Result as IoResult, Write};

/// A [`Write`] wrapper that accumulates a CRC-32 over everything written.
///
/// All writes pass through to the inner writer unchanged; call
/// [`CrcWriter::checksum`] after writing the payload to obtain the CRC to store
/// (e.g. in a footer), and [`CrcWriter::bytes_written`] for the payload length.
pub struct CrcWriter<W: Write> {
    inner: W,
    hasher: crc32fast::Hasher,
    bytes_written: u64,
}

impl<W: Write> CrcWriter<W> {
    /// Wrap `inner`, starting from a zero checksum and byte count.
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            bytes_written: 0,
        }
    }

    /// The CRC-32 of all bytes written so far.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// The number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Consume the wrapper and return the inner writer.
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for CrcWriter<W> {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes_written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.inner.flush()
    }
}

/// A [`Read`] wrapper that accumulates a CRC-32 over everything read.
///
/// All reads pass through from the inner reader unchanged; call
/// [`CrcReader::checksum`] after reading the payload to compare against a
/// stored CRC, and [`CrcReader::bytes_read`] to learn how many payload bytes
/// were consumed (e.g. to locate a trailing footer via the file size).
pub struct CrcReader<R: Read> {
    inner: R,
    hasher: crc32fast::Hasher,
    bytes_read: u64,
}

impl<R: Read> CrcReader<R> {
    /// Wrap `inner`, starting from a zero checksum and byte count.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: crc32fast::Hasher::new(),
            bytes_read: 0,
        }
    }

    /// The CRC-32 of all bytes read so far.
    pub fn checksum(&self) -> u32 {
        self.hasher.clone().finalize()
    }

    /// The number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Consume the wrapper and return the inner reader.
    pub fn into_inner(self) -> R {
        self.inner
    }

    /// Borrow the inner reader (e.g. to read a footer that must not be
    /// included in the running checksum).
    pub fn get_mut(&mut self) -> &mut R {
        &mut self.inner
    }
}

impl<R: Read> Read for CrcReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let n = self.inner.read(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes_read += n as u64;
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn writer_checksum_matches_standalone_hash() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        assert_eq!(w.bytes_written(), data.len() as u64);

        let mut h = crc32fast::Hasher::new();
        h.update(data);
        assert_eq!(w.checksum(), h.finalize());
    }

    #[test]
    fn reader_checksum_matches_writer() {
        let data = b"laurus integrity check payload";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        let expected = w.checksum();
        let buf = w.into_inner();

        let mut r = CrcReader::new(Cursor::new(buf));
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, data);
        assert_eq!(r.bytes_read(), data.len() as u64);
        assert_eq!(r.checksum(), expected);
    }

    #[test]
    fn reader_detects_corruption() {
        let data = b"a payload that will be corrupted";
        let mut w = CrcWriter::new(Vec::new());
        w.write_all(data).unwrap();
        let good = w.checksum();
        let mut buf = w.into_inner();
        buf[5] ^= 0xff; // flip a bit

        let mut r = CrcReader::new(Cursor::new(buf));
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_ne!(r.checksum(), good, "a corrupted byte must change the CRC");
    }
}
