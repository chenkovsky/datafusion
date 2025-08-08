// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{
    Array, ArrayAccessor, ArrayIter, ArrayRef, ArrowPrimitiveType, BinaryViewArray, BooleanArray, Date32Array, Date64Array, Decimal128Array, Decimal256Array, DurationMicrosecondArray, DurationMillisecondArray, DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray, Float32Array, Float64Array, GenericBinaryArray, GenericByteArray, Int16Array, Int32Array, Int64Array, Int8Array, IntervalDayTimeArray, IntervalMonthDayNanoArray, IntervalYearMonthArray, LargeStringArray, ListArray, OffsetSizeTrait, PrimitiveArray, StringArray, StringViewArray, Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array
};
use arrow::compute;
use arrow::datatypes::{i256, ByteArrayType, DataType, GenericBinaryType, GenericStringType, Int64Type, IntervalDayTime, IntervalMonthDayNano, IntervalUnit, TimeUnit};
use bigdecimal::num_traits::{Float, ToBytes};
use datafusion_common::{
    exec_err, Result,
};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;

use std::any::Any;
use std::sync::Arc;
use twox_hash::XxHash64;

/// <https://spark.apache.org/docs/latest/api/sql/index.html#xxhash64>
#[derive(Debug)]
pub struct XxHash64Func {
    signature: Signature,
    seed: u64,
}

impl Default for XxHash64Func {
    fn default() -> Self {
        Self::new(42)
    }
}

impl XxHash64Func {
    pub fn new(seed: u64) -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            seed,
        }
    }
}

impl ScalarUDFImpl for XxHash64Func {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "xxhash64"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let func = |arr: &[ArrayRef]| spark_xxhash64(arr, self.seed);
        make_scalar_function(func, vec![])(&args.args)
    }
}

fn xxhash_boolean(arr: &BooleanArray, seed: u64) -> ArrayRef
{
    let true_hash = XxHash64::oneshot(seed, 1_i32.to_le_bytes().as_ref()) as i64;
    let false_hash = XxHash64::oneshot(seed, 0_i32.to_le_bytes().as_ref()) as i64;
    let arr = arr.iter().map(|value| {
        value.map(|v| 
            if v { true_hash } else { false_hash }
        )
    }).collect::<PrimitiveArray<Int64Type>>();
    Arc::new(arr)
}

fn xxhash_primitive<T, U>(arr: &PrimitiveArray<T>, seed: u64) -> ArrayRef
where
    T: ArrowPrimitiveType,
    T::Native: Into<U>,
    U: ToBytes,
{
    Arc::new(compute::unary::<_, _, Int64Type>(arr, |value| {
        let bytes = (value.into() as U).to_le_bytes();
        XxHash64::oneshot(seed, bytes.as_ref()) as i64
    }))
}

fn xxhash_primitive_float<T, U>(arr: &PrimitiveArray<T>, seed: u64) -> ArrayRef
where
    T: ArrowPrimitiveType,
    T::Native: ToBytes + Float,
    U: ToBytes + Default,
{
    let neg_zero = T::Native::neg_zero();
    let neg_zero_bytes = U::to_le_bytes(&U::default());
    Arc::new(compute::unary::<_, _, Int64Type>(arr, |value| {
        if value == neg_zero {
            XxHash64::oneshot(seed, neg_zero_bytes.as_ref()) as i64
        } else {
            let bytes = value.to_le_bytes();
            XxHash64::oneshot(seed, bytes.as_ref()) as i64
        }
    }))
}

fn xxhash_bytes<T>(iter: ArrayIter<T>, seed: u64) -> ArrayRef
where
    T: ArrayAccessor,
    T::Item: AsRef<[u8]>,
{
    let arr = iter.map(|value| {
        value.map(|v| {
            XxHash64::oneshot(seed, v.as_ref()) as i64
        })
    }).collect::<PrimitiveArray<Int64Type>>();
    Arc::new(arr)
}

struct I256(i256);

impl From<i256> for I256 {
    fn from(value: i256) -> Self {
        I256(value)
    }
}

impl ToBytes for I256 {
    type Bytes = [u8; 32];

    fn to_le_bytes(&self) -> Self::Bytes {
        self.0.to_le_bytes()
    }

    fn to_be_bytes(&self) -> Self::Bytes {
        self.0.to_be_bytes()
    }
}

struct _IntervalDayTime(IntervalDayTime);

impl From<IntervalDayTime> for _IntervalDayTime {
    fn from(value: IntervalDayTime) -> Self {
        _IntervalDayTime(value)
    }
}

impl ToBytes for _IntervalDayTime {
    type Bytes = [u8; 8];

    fn to_le_bytes(&self) -> Self::Bytes {
        let days = self.0.days.to_le_bytes();
        let milliseconds = self.0.milliseconds.to_le_bytes();
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&days);
        bytes[4..].copy_from_slice(&milliseconds);
        bytes
    }

    fn to_be_bytes(&self) -> Self::Bytes {
        let days = self.0.days.to_be_bytes();
        let milliseconds = self.0.milliseconds.to_be_bytes();
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&days);
        bytes[4..].copy_from_slice(&milliseconds);
        bytes
    }
}

struct _IntervalMonthDayNano(IntervalMonthDayNano);

impl From<IntervalMonthDayNano> for _IntervalMonthDayNano {
    fn from(value: IntervalMonthDayNano) -> Self {
        _IntervalMonthDayNano(value)
    }
}

impl ToBytes for _IntervalMonthDayNano {
    type Bytes = [u8; 16];

    fn to_le_bytes(&self) -> Self::Bytes {
        let months = self.0.months.to_le_bytes();
        let days = self.0.days.to_le_bytes();
        let nanoseconds = self.0.nanoseconds.to_le_bytes();
        let mut bytes = [0; 16];
        bytes[..4].copy_from_slice(&months);
        bytes[4..8].copy_from_slice(&days);
        bytes[8..].copy_from_slice(&nanoseconds);
        bytes
    }

    fn to_be_bytes(&self) -> Self::Bytes { 
        let months = self.0.months.to_be_bytes();
        let days = self.0.days.to_be_bytes();
        let nanoseconds = self.0.nanoseconds.to_be_bytes();
        let mut bytes = [0; 16];
        bytes[..4].copy_from_slice(&months);
        bytes[4..8].copy_from_slice(&days);
        bytes[8..].copy_from_slice(&nanoseconds);
        bytes
    }
}

/// Computes the xxHash64 hash of the given data
pub fn spark_xxhash64(data: &[ArrayRef], seed: u64) -> Result<ArrayRef> {
    let arr = data[0];
    let x = match data[0].data_type() {
        DataType::Boolean => {
            let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            xxhash_boolean(arr, seed)
        }
        DataType::Int8 => {
            let arr = arr.as_any().downcast_ref::<Int8Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::UInt8 => {
            let arr = arr.as_any().downcast_ref::<UInt8Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Int16 => {
            let arr = arr.as_any().downcast_ref::<Int16Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::UInt16 => {
            let arr = arr.as_any().downcast_ref::<UInt16Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Int32 => {
            let arr = arr.as_any().downcast_ref::<Int32Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::UInt32 => {
            let arr = arr.as_any().downcast_ref::<UInt32Array>().unwrap();
            xxhash_primitive::<_, u32>(arr, seed)
        }
        DataType::Int64 => {
            let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::UInt64 => {
            let arr = arr.as_any().downcast_ref::<UInt64Array>().unwrap();
            xxhash_primitive::<_, u64>(arr, seed)
        }
        DataType::Float32 => {
            let arr = arr.as_any().downcast_ref::<Float32Array>().unwrap();
            xxhash_primitive_float::<_, i32>(arr, seed)
        }
        DataType::Float64 => {
            let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            xxhash_primitive_float::<_, i64>(arr, seed)
        }
        DataType::Decimal128(precision, scale) => {
            let arr = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
            xxhash_primitive::<_, i128>(arr, seed)
        }
        DataType::Decimal256(precision, scale) => {
            let arr = arr.as_any().downcast_ref::<Decimal256Array>().unwrap();
            xxhash_primitive::<_, I256>(arr, seed)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            let arr = arr.as_any().downcast_ref::<TimestampSecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            let arr = arr.as_any().downcast_ref::<TimestampMillisecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            let arr = arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            let arr = arr.as_any().downcast_ref::<TimestampNanosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Date32 => {
            let arr = arr.as_any().downcast_ref::<Date32Array>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Date64 => {
            let arr = arr.as_any().downcast_ref::<Date64Array>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Utf8 => {
            let arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::LargeUtf8 => {
            let arr = arr.as_any().downcast_ref::<LargeStringArray>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::Binary => {
            let arr = arr.as_any().downcast_ref::<GenericBinaryArray<i32>>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::LargeBinary => {
            let arr = arr.as_any().downcast_ref::<GenericBinaryArray<i64>>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::FixedSizeBinary(_) => {
            let arr = arr.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::Utf8View => {
            let arr = arr.as_any().downcast_ref::<StringViewArray>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::BinaryView => {
            let arr = arr.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            xxhash_bytes(arr.iter(), seed)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            let arr = arr.as_any().downcast_ref::<IntervalDayTimeArray>().unwrap();
            xxhash_primitive::<_, _IntervalDayTime>(arr, seed)
        }
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let arr = arr.as_any().downcast_ref::<IntervalMonthDayNanoArray>().unwrap();
            xxhash_primitive::<_, _IntervalMonthDayNano>(arr, seed)
        }
        DataType::Interval(IntervalUnit::YearMonth) => {
            let arr = arr.as_any().downcast_ref::<IntervalYearMonthArray>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Duration(TimeUnit::Second) => {
            let arr = arr.as_any().downcast_ref::<DurationSecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Duration(TimeUnit::Millisecond) => {
            let arr = arr.as_any().downcast_ref::<DurationMillisecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            let arr = arr.as_any().downcast_ref::<DurationMicrosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Duration(TimeUnit::Nanosecond) => {
            let arr = arr.as_any().downcast_ref::<DurationNanosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Time32(TimeUnit::Second) => {
            let arr = arr.as_any().downcast_ref::<Time32SecondArray>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            let arr = arr.as_any().downcast_ref::<Time32MillisecondArray>().unwrap();
            xxhash_primitive::<_, i32>(arr, seed)
        }
        DataType::Time32(TimeUnit::Nanosecond) => {
            let arr = arr.as_any().downcast_ref::<Time64NanosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            let arr = arr.as_any().downcast_ref::<Time64MicrosecondArray>().unwrap();
            xxhash_primitive::<_, i64>(arr, seed)
        }
        DataType::List(_) => {
            let arr = arr.as_any().downcast_ref::<ListArray>().unwrap();
        }
        _ => {
            return exec_err!("Unsupported data type: {}", data[0].data_type());
        }
    };
}
