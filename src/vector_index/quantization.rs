//! Vector quantization for memory-efficient storage.

use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::vector::Vector;

/// Quantization methods for compressing vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantizationMethod {
    /// No quantization.
    #[default]
    None,
    /// Scalar quantization to 8-bit integers.
    Scalar8Bit,
    /// Product quantization.
    ProductQuantization { subvector_count: usize },
}

/// Vector quantizer for compressing and decompressing vectors.
pub struct VectorQuantizer {
    method: QuantizationMethod,
    dimension: usize,
    is_trained: bool,
    // Scalar quantization parameters
    min_values: Option<Vec<f32>>,
    max_values: Option<Vec<f32>>,
    // Product quantization parameters
    codebooks: Option<Vec<Vec<Vec<f32>>>>,
}

impl VectorQuantizer {
    /// Create a new vector quantizer.
    pub fn new(method: QuantizationMethod, dimension: usize) -> Self {
        Self {
            method,
            dimension,
            is_trained: false,
            min_values: None,
            max_values: None,
            codebooks: None,
        }
    }

    /// Train the quantizer on a set of vectors.
    pub fn train(&mut self, vectors: &[Vector]) -> Result<()> {
        match self.method {
            QuantizationMethod::None => {
                // No training needed
            }
            QuantizationMethod::Scalar8Bit => {
                self.train_scalar_quantization(vectors)?;
            }
            QuantizationMethod::ProductQuantization { subvector_count } => {
                self.train_product_quantization(vectors, subvector_count)?;
            }
        }

        self.is_trained = true;
        Ok(())
    }

    /// Quantize a vector to compressed representation.
    pub fn quantize(&self, vector: &Vector) -> Result<QuantizedVector> {
        if !self.is_trained && self.method != QuantizationMethod::None {
            return Err(SageError::InvalidOperation(
                "Quantizer must be trained before use".to_string(),
            ));
        }

        if vector.dimension() != self.dimension {
            return Err(SageError::InvalidOperation(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.dimension()
            )));
        }

        match self.method {
            QuantizationMethod::None => Ok(QuantizedVector {
                method: self.method,
                data: QuantizedData::Float32(vector.data.clone()),
            }),
            QuantizationMethod::Scalar8Bit => {
                let quantized_data = self.scalar_quantize(&vector.data)?;
                Ok(QuantizedVector {
                    method: self.method,
                    data: QuantizedData::Uint8(quantized_data),
                })
            }
            QuantizationMethod::ProductQuantization { .. } => {
                let quantized_data = self.product_quantize(&vector.data)?;
                Ok(QuantizedVector {
                    method: self.method,
                    data: QuantizedData::ProductCodes(quantized_data),
                })
            }
        }
    }

    /// Dequantize a compressed vector back to full precision.
    pub fn dequantize(&self, quantized: &QuantizedVector) -> Result<Vector> {
        if quantized.method != self.method {
            return Err(SageError::InvalidOperation(
                "Quantization method mismatch".to_string(),
            ));
        }

        let data = match &quantized.data {
            QuantizedData::Float32(data) => data.clone(),
            QuantizedData::Uint8(data) => self.scalar_dequantize(data)?,
            QuantizedData::ProductCodes(codes) => self.product_dequantize(codes)?,
        };

        Ok(Vector::new(data))
    }

    /// Train scalar quantization by finding min/max values per dimension.
    fn train_scalar_quantization(&mut self, vectors: &[Vector]) -> Result<()> {
        if vectors.is_empty() {
            return Err(SageError::InvalidOperation(
                "Cannot train on empty vector set".to_string(),
            ));
        }

        let mut min_values = vec![f32::INFINITY; self.dimension];
        let mut max_values = vec![f32::NEG_INFINITY; self.dimension];

        for vector in vectors {
            for (i, &value) in vector.data.iter().enumerate() {
                if i < self.dimension {
                    min_values[i] = min_values[i].min(value);
                    max_values[i] = max_values[i].max(value);
                }
            }
        }

        self.min_values = Some(min_values);
        self.max_values = Some(max_values);
        Ok(())
    }

    /// Perform scalar quantization to 8-bit integers.
    fn scalar_quantize(&self, data: &[f32]) -> Result<Vec<u8>> {
        let min_values = self.min_values.as_ref().unwrap();
        let max_values = self.max_values.as_ref().unwrap();

        let quantized: Vec<u8> = data
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let min_val = min_values[i];
                let max_val = max_values[i];
                let range = max_val - min_val;

                if range > 0.0 {
                    let normalized = (value - min_val) / range;
                    (normalized * 255.0).clamp(0.0, 255.0) as u8
                } else {
                    0
                }
            })
            .collect();

        Ok(quantized)
    }

    /// Dequantize 8-bit integers back to float32.
    fn scalar_dequantize(&self, data: &[u8]) -> Result<Vec<f32>> {
        let min_values = self.min_values.as_ref().unwrap();
        let max_values = self.max_values.as_ref().unwrap();

        let dequantized: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let min_val = min_values[i];
                let max_val = max_values[i];
                let range = max_val - min_val;

                let normalized = value as f32 / 255.0;
                min_val + normalized * range
            })
            .collect();

        Ok(dequantized)
    }

    /// Train product quantization (placeholder implementation).
    fn train_product_quantization(
        &mut self,
        _vectors: &[Vector],
        _subvector_count: usize,
    ) -> Result<()> {
        // This would implement k-means clustering for each subvector
        // For now, just a placeholder
        self.codebooks = Some(vec![vec![vec![0.0; self.dimension / 4]; 256]; 4]);
        Ok(())
    }

    /// Perform product quantization (placeholder implementation).
    fn product_quantize(&self, _data: &[f32]) -> Result<Vec<u8>> {
        // This would find the nearest codebook entry for each subvector
        // For now, just a placeholder
        Ok(vec![0; 4])
    }

    /// Dequantize product codes (placeholder implementation).
    fn product_dequantize(&self, _codes: &[u8]) -> Result<Vec<f32>> {
        // This would reconstruct the vector from codebook entries
        // For now, just a placeholder
        Ok(vec![0.0; self.dimension])
    }

    /// Get the compression ratio achieved by this quantization method.
    pub fn compression_ratio(&self) -> f32 {
        match self.method {
            QuantizationMethod::None => 1.0,
            QuantizationMethod::Scalar8Bit => 4.0, // 32-bit to 8-bit
            QuantizationMethod::ProductQuantization { subvector_count } => {
                (self.dimension * 4) as f32 / subvector_count as f32
            }
        }
    }

    /// Check if the quantizer is trained.
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

/// Compressed vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedVector {
    method: QuantizationMethod,
    data: QuantizedData,
}

/// Different types of quantized data.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum QuantizedData {
    Float32(Vec<f32>),
    Uint8(Vec<u8>),
    ProductCodes(Vec<u8>),
}

impl QuantizedVector {
    /// Get the memory size of this quantized vector in bytes.
    pub fn memory_size(&self) -> usize {
        match &self.data {
            QuantizedData::Float32(data) => data.len() * 4,
            QuantizedData::Uint8(data) => data.len(),
            QuantizedData::ProductCodes(data) => data.len(),
        }
    }

    /// Get the quantization method used.
    pub fn method(&self) -> QuantizationMethod {
        self.method
    }
}
