//! Conversions between Python objects and Laurus types.

use chrono::{DateTime, Utc};
use laurus::{DataValue, Document};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};

/// Convert a Python `dict` to a [`Document`].
pub fn dict_to_document(py: Python, dict: &Bound<PyDict>) -> PyResult<Document> {
    let mut builder = Document::builder();
    for (key, value) in dict.iter() {
        let field: String = key.extract()?;
        let dv = py_to_data_value(py, &value)?;
        builder = builder.add_field(&field, dv);
    }
    Ok(builder.build())
}

/// Convert a Python value to a [`DataValue`].
///
/// Type mapping:
/// - `None`                → `DataValue::Null`
/// - `bool`                → `DataValue::Bool`  (must be checked before int)
/// - `int`                 → `DataValue::Int64`
/// - `float`               → `DataValue::Float64`
/// - `str`                 → `DataValue::Text`
/// - `bytes`               → `DataValue::Bytes`
/// - `list[float|int]`     → `DataValue::Vector`
/// - `(lat, lon)` tuple    → `DataValue::Geo`
/// - `(x, y, z)` tuple     → `DataValue::GeoEcef` (3D ECEF Cartesian, meters)
pub fn py_to_data_value(_py: Python, obj: &Bound<PyAny>) -> PyResult<DataValue> {
    if obj.is_none() {
        return Ok(DataValue::Null);
    }
    // bool must come before int because Python bool is a subclass of int
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(DataValue::Bool(b));
    }
    if obj.is_instance_of::<PyInt>() {
        let i: i64 = obj.extract()?;
        return Ok(DataValue::Int64(i));
    }
    if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        return Ok(DataValue::Float64(f));
    }
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        return Ok(DataValue::Text(s));
    }
    if obj.is_instance_of::<PyBytes>() {
        let b: Vec<u8> = obj.extract()?;
        return Ok(DataValue::Bytes(b, None));
    }
    if obj.is_instance_of::<PyList>() {
        let list = obj.cast::<PyList>()?;
        let vec: Vec<f32> = list
            .iter()
            .map(|item| item.extract::<f32>())
            .collect::<PyResult<_>>()?;
        return Ok(DataValue::Vector(vec));
    }
    // Try tuple (lat, lon) for Geo
    if let Ok(tup) = obj.cast::<pyo3::types::PyTuple>()
        && tup.len() == 2
        && let (Ok(lat), Ok(lon)) = (
            tup.get_item(0)?.extract::<f64>(),
            tup.get_item(1)?.extract::<f64>(),
        )
    {
        let point = laurus::lexical::GeoPoint::try_new(lat, lon)
            .map_err(|e| PyValueError::new_err(format!("invalid geo point: {e}")))?;
        return Ok(DataValue::Geo(point));
    }
    // Try tuple (x, y, z) for Geo3d (ECEF Cartesian, meters). Must come
    // after the 2-tuple Geo check so existing semantics are preserved.
    if let Ok(tup) = obj.cast::<pyo3::types::PyTuple>()
        && tup.len() == 3
        && let (Ok(x), Ok(y), Ok(z)) = (
            tup.get_item(0)?.extract::<f64>(),
            tup.get_item(1)?.extract::<f64>(),
            tup.get_item(2)?.extract::<f64>(),
        )
    {
        return Ok(DataValue::GeoEcef(laurus::GeoEcefPoint::new(x, y, z)));
    }
    // Try Python datetime.datetime
    if let Ok(dt_str) = obj.call_method0("isoformat")
        && let Ok(s) = dt_str.extract::<String>()
    {
        if let Ok(dt) = s.parse::<DateTime<Utc>>() {
            return Ok(DataValue::DateTime(dt));
        }
        // Try without timezone suffix
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%dT%H:%M:%S")
            .map(|ndt| DateTime::<Utc>::from_naive_utc_and_offset(ndt, Utc))
        {
            return Ok(DataValue::DateTime(dt));
        }
    }
    Err(PyValueError::new_err(format!(
        "Cannot convert Python value of type {} to DataValue",
        obj.get_type().name()?
    )))
}

/// Convert a [`Document`] to a Python `dict`.
pub fn document_to_dict(py: Python, doc: &Document) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (field, value) in &doc.fields {
        let py_value = data_value_to_py(py, value)?;
        dict.set_item(field, py_value)?;
    }
    Ok(dict.into_any().unbind())
}

/// Convert a [`DataValue`] to a Python object.
pub fn data_value_to_py(py: Python, value: &DataValue) -> PyResult<Py<PyAny>> {
    match value {
        DataValue::Null => Ok(py.None()),
        DataValue::Bool(b) => Ok((*(*b).into_pyobject(py)?).clone().unbind().into_any()),
        DataValue::Int64(i) => Ok((*i).into_pyobject(py)?.unbind().into_any()),
        DataValue::Float64(f) => Ok((*f).into_pyobject(py)?.unbind().into_any()),
        DataValue::Text(s) => Ok(s.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::Bytes(b, _mime) => Ok(PyBytes::new(py, b).unbind().into_any()),
        DataValue::Vector(v) => Ok(v.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::DateTime(dt) => Ok(dt.to_rfc3339().into_pyobject(py)?.unbind().into_any()),
        DataValue::Geo(p) => {
            let tup = pyo3::types::PyTuple::new(py, [p.lat, p.lon])?;
            Ok(tup.unbind().into_any())
        }
        DataValue::GeoEcef(p) => {
            let tup = pyo3::types::PyTuple::new(py, [p.x, p.y, p.z])?;
            Ok(tup.unbind().into_any())
        }
        DataValue::Int64Array(arr) => Ok(arr.clone().into_pyobject(py)?.unbind().into_any()),
        DataValue::Float64Array(arr) => Ok(arr.clone().into_pyobject(py)?.unbind().into_any()),
    }
}
