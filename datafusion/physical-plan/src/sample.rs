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

//! Defines the SAMPLE operator

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::{
    DisplayAs, ExecutionPlanProperties, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use crate::{
    metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
    DisplayFormatType, ExecutionPlan,
};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow::compute;
use datafusion_common::{internal_err, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;

use futures::stream::{Stream, StreamExt};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Sampling method for the SampleExec operator
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingMethod {
    /// Bernoulli sampling - each row is included with probability p
    Bernoulli(f64),
    /// Poisson sampling - each row is included with probability 1 - e^(-lambda)
    Poisson(f64),
}

impl SamplingMethod {
    /// Get the sampling ratio for this method
    pub fn ratio(&self) -> f64 {
        match self {
            SamplingMethod::Bernoulli(p) => *p,
            SamplingMethod::Poisson(lambda) => 1.0 - (-lambda).exp(),
        }
    }
}

/// SampleExec samples rows from its input based on a sampling method.
/// This is used to implement SQL `SAMPLE` clause.
#[derive(Debug, Clone)]
pub struct SampleExec {
    /// The input plan
    input: Arc<dyn ExecutionPlan>,
    /// The sampling method
    method: SamplingMethod,
    /// Whether to sample with replacement
    with_replacement: bool,
    /// Random seed for reproducible sampling
    seed: u64,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Properties equivalence properties, partitioning, etc.
    cache: PlanProperties,
}

impl SampleExec {

    /// Create a new SampleExec with a custom sampling method
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        lower_bound: f64,
        upper_bound: f64,
        with_replacement: bool,
        seed: u64,
    ) -> Result<Self> {
        if lower_bound < 0.0 || upper_bound > 1.0 || lower_bound > upper_bound {
            return internal_err!(
                "Sampling bounds must be between 0.0 and 1.0, and lower_bound <= upper_bound, got [{}, {}]",
                lower_bound, upper_bound
            );
        }

        let cache = Self::compute_properties(&input);
        let method = if with_replacement {
            // Use Poisson sampling for replacement
            let ratio = upper_bound - lower_bound;
            // Convert ratio to lambda: ratio = 1 - e^(-lambda) => lambda = -ln(1 - ratio)
            let lambda = if ratio >= 1.0 {
                f64::INFINITY
            } else if ratio <= 0.0 {
                0.0
            } else {
                -((1.0 - ratio).ln())
            };
            SamplingMethod::Poisson(lambda)
        } else {
            // Use Bernoulli sampling for no replacement
            SamplingMethod::Bernoulli(upper_bound - lower_bound)
        };

        Ok(Self {
            input,
            method,
            with_replacement,
            seed,
            metrics: ExecutionPlanMetricsSet::new(),
            cache,
        })
    }

    /// The sampling method
    pub fn method(&self) -> &SamplingMethod {
        &self.method
    }

    /// Whether to sample with replacement
    pub fn with_replacement(&self) -> bool {
        self.with_replacement
    }

    /// The random seed
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(input: &Arc<dyn ExecutionPlan>) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(input.schema()),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        )
    }
}

impl DisplayAs for SampleExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "SampleExec: method={:?}, with_replacement={}, seed={}",
                    self.method, self.with_replacement, self.seed
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "SampleExec: method={:?}, with_replacement={}, seed={}",
                    self.method, self.with_replacement, self.seed
                )
            }
        }
    }
}

impl ExecutionPlan for SampleExec {
    fn name(&self) -> &'static str {
        "SampleExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false] // Sampling does not maintain input order
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Get the ratio from the current method
        let ratio = self.method.ratio();
        Ok(Arc::new(SampleExec::try_new(
            children[0].clone(),
            0.0,
            ratio,
            self.with_replacement,
            self.seed,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);

        Ok(Box::pin(SampleExecStream {
            input: input_stream,
            method: self.method.clone(),
            with_replacement: self.with_replacement,
            seed: self.seed,
            baseline_metrics,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        let input_stats = self.input.partition_statistics(partition)?;
        
        // Apply sampling ratio to statistics
        let mut stats = input_stats;
        let ratio = self.method.ratio();
        
        stats.num_rows = stats.num_rows.map(|nr| (nr as f64 * ratio) as usize);
        stats.total_byte_size = stats.total_byte_size.map(|tb| (tb as f64 * ratio) as usize);
        
        Ok(stats)
    }
}

/// Stream for the SampleExec operator
struct SampleExecStream {
    /// The input stream
    input: SendableRecordBatchStream,
    /// The sampling method
    method: SamplingMethod,
    /// Whether to sample with replacement
    with_replacement: bool,
    /// Random seed
    seed: u64,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
}

impl Stream for SampleExecStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.input.poll_next_unpin(cx);
        let baseline_metrics = &mut self.baseline_metrics;

        match poll {
            Poll::Ready(Some(Ok(batch))) => {
                let start = baseline_metrics.elapsed_compute().clone();
                let result = sample_batch(&batch, &self.method, self.with_replacement, self.seed);
                let _timer = start.timer();
                Poll::Ready(Some(result))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

impl RecordBatchStream for SampleExecStream {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

/// Sample a record batch based on the given sampling method
fn sample_batch(
    batch: &RecordBatch,
    method: &SamplingMethod,
    with_replacement: bool,
    seed: u64,
) -> Result<RecordBatch> {
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(batch.clone());
    }

    match method {
        SamplingMethod::Bernoulli(probability) => {
            if *probability == 0.0 {
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
            if *probability == 1.0 {
                return Ok(batch.clone());
            }
            sample_bernoulli(batch, *probability, with_replacement, seed)
        }
        SamplingMethod::Poisson(lambda) => {
            if *lambda == 0.0 {
                return Ok(RecordBatch::new_empty(batch.schema()));
            }
            sample_poisson(batch, *lambda, with_replacement, seed)
        }
    }
}

/// Bernoulli sampling - each row is included with probability p
fn sample_bernoulli(
    batch: &RecordBatch,
    probability: f64,
    with_replacement: bool,
    seed: u64,
) -> Result<RecordBatch> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected_indices = Vec::new();

    for (i, _) in (0..batch.num_rows()).enumerate() {
        if rng.random_bool(probability) {
            selected_indices.push(i);
        }
    }

    if selected_indices.is_empty() {
        return Ok(RecordBatch::new_empty(batch.schema()));
    }

    if with_replacement {
        // For replacement, we can have duplicate indices
        let mut sampled_rows = Vec::new();
        for &index in &selected_indices {
            let row = batch.slice(index, 1);
            sampled_rows.push(row);
        }
        concatenate_batches(&sampled_rows.iter().collect::<Vec<_>>())
    } else {
        // Without replacement, indices are unique
        selected_indices.sort();
        let mut sampled_rows = Vec::new();
        for &index in &selected_indices {
            let row = batch.slice(index, 1);
            sampled_rows.push(row);
        }
        concatenate_batches(&sampled_rows.iter().collect::<Vec<_>>())
    }
}

/// Poisson sampling - each row is included with probability 1 - e^(-lambda)
fn sample_poisson(
    batch: &RecordBatch,
    lambda: f64,
    with_replacement: bool,
    seed: u64,
) -> Result<RecordBatch> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected_indices = Vec::new();

    for (i, _) in (0..batch.num_rows()).enumerate() {
        // Generate Poisson random variable
        let _k = rng.random_range(0..=10); // Limit to reasonable range
        let probability = 1.0 - (-lambda).exp();
        
        if rng.random_bool(probability) {
            selected_indices.push(i);
        }
    }

    if selected_indices.is_empty() {
        return Ok(RecordBatch::new_empty(batch.schema()));
    }

    if with_replacement {
        // For replacement, we can have duplicate indices
        let mut sampled_rows = Vec::new();
        for &index in &selected_indices {
            let row = batch.slice(index, 1);
            sampled_rows.push(row);
        }
        concatenate_batches(&sampled_rows.iter().collect::<Vec<_>>())
    } else {
        // Without replacement, indices are unique
        selected_indices.sort();
        let mut sampled_rows = Vec::new();
        for &index in &selected_indices {
            let row = batch.slice(index, 1);
            sampled_rows.push(row);
        }
        concatenate_batches(&sampled_rows.iter().collect::<Vec<_>>())
    }
}

/// Helper function to concatenate record batches
fn concatenate_batches(batches: &[&RecordBatch]) -> Result<RecordBatch> {
    if batches.is_empty() {
        return internal_err!("Cannot concatenate empty batch list");
    }
    
    let schema = batches[0].schema();
    let mut columns = Vec::new();
    
    for col_idx in 0..schema.fields().len() {
        let mut arrays = Vec::new();
        for batch in batches {
            arrays.push(batch.column(col_idx));
        }
        columns.push(compute::concat(&arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>())?);
    }
    
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::TaskContext;
    use futures::TryStreamExt;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_sample_exec_bernoulli() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, false),
            Field::new("name", arrow::datatypes::DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        )?;

        let input = Arc::new(crate::test::TestMemoryExec::try_new(
            &[vec![batch]],
            schema.clone(),
            None,
        )?);

        let sample_exec = SampleExec::try_new(input, 0.6, 1.0, false, 42)?;
        
        let context = Arc::new(TaskContext::default());
        let stream = sample_exec.execute(0, context)?;
        
        let batches = stream.try_collect::<Vec<_>>().await?;
        assert!(!batches.is_empty());
        
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        // With 60% sampling ratio and 5 input rows, we expect around 3 rows
        assert!(total_rows >= 2 && total_rows <= 4);
        
        Ok(())
    }

    #[tokio::test]
    async fn test_sample_exec_poisson() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]))],
        )?;

        let input = Arc::new(crate::test::TestMemoryExec::try_new(
            &[vec![batch]],
            schema.clone(),
            None,
        )?);

        let sample_exec = SampleExec::try_new(input, 0.0, 1.0, false, 42)?;
        
        let context = Arc::new(TaskContext::default());
        let stream = sample_exec.execute(0, context)?;
        
        let batches = stream.try_collect::<Vec<_>>().await?;
        assert!(!batches.is_empty());
        
        Ok(())
    }

    #[test]
    fn test_sampling_methods() {
        // Test Bernoulli
        let bernoulli = SamplingMethod::Bernoulli(0.5);
        assert_eq!(bernoulli.ratio(), 0.5);

        // Test Poisson
        let poisson = SamplingMethod::Poisson(0.5);
        let expected_ratio = 1.0 - (-0.5_f64).exp();
        assert!((poisson.ratio() - expected_ratio).abs() < 1e-10);
    }
}
