use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow_schema::SchemaRef;
use datafusion_common::stats::Precision;
use datafusion_physical_expr::equivalence::ProjectionMapping;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{calculate_union, Partitioning, PhysicalExpr};
use datafusion_physical_expr_common::physical_expr::fmt_sql;
use itertools::Itertools;

use super::{
    DisplayAs, ExecutionPlanProperties, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream, Statistics,
};
use datafusion_common::{ColumnStatistics, DataFusionError, Result};
use futures::stream::{Stream, StreamExt};

use crate::execution_plan::ExecutionPlan;
use crate::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use crate::DisplayFormatType;

/**
 * Apply a number of projections to every input row, hence we will get multiple output rows for
 * an input row.
 */
#[derive(Debug, Clone)]
pub struct ExpandExec {
    exprs: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
}

impl ExpandExec {
    pub fn try_new_with_schema(
        exprs: Vec<Vec<Arc<dyn PhysicalExpr>>>,
        input: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let metrics = ExecutionPlanMetricsSet::new();
        let cache = Self::compute_properties(&input, &exprs, Arc::clone(&schema))?;
        Ok(Self {
            exprs,
            input,
            schema,
            metrics,
            cache,
        })
    }

    pub fn exprs(&self) -> &Vec<Vec<Arc<dyn PhysicalExpr>>> {
        &self.exprs
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    fn compute_properties(
        input: &Arc<dyn ExecutionPlan>,
        exprs: &Vec<Vec<Arc<dyn PhysicalExpr>>>,
        schema: SchemaRef,
    ) -> Result<PlanProperties> {
        let input_partition = input.output_partitioning();
        let mut eq_properties = Vec::new();
        let mut output_partitioning = None;
        for row_expr in exprs {
            let expr = row_expr
                .iter()
                .zip(schema.fields().iter())
                .map(|(expr, field)| (Arc::clone(expr), field.name().clone()))
                .collect::<Vec<_>>();
            let projection_mapping =
                ProjectionMapping::try_new(expr.as_slice(), &input.schema())?;
            let mut input_eq_properties = input.equivalence_properties().clone();
            input_eq_properties.substitute_oeq_class(&projection_mapping)?;
            eq_properties
                .push(input_eq_properties.project(&projection_mapping, Arc::clone(&schema)));

            let partitioning =
                input_partition.project(&projection_mapping, &input_eq_properties);
            match output_partitioning {
                Some(p @ Partitioning::Hash(..)) if p != partitioning => {
                    output_partitioning = Some(Partitioning::UnknownPartitioning(
                        input_partition.partition_count(),
                    ));
                }
                None => {
                    output_partitioning = Some(partitioning);
                }
                _ => {}
            }
        }
        let eq_properties = calculate_union(eq_properties, schema)?;
        let output_partitioning = output_partitioning.unwrap_or(
            Partitioning::UnknownPartitioning(input_partition.partition_count()),
        );
        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        ))
    }
}

impl DisplayAs for ExpandExec {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let exprs = self.exprs
                        .iter()
                        .map(|lst| {
                            format!("[{}]", lst.iter().map(|e| e.to_string()).join(", "))
                        })
                        .join(",").to_string();
                write!(f, "ExpandExec: exprs=[{}]", exprs)
            }
            DisplayFormatType::TreeRender => {
                for (i, exprs) in self.exprs().iter().enumerate() {
                    for (j, expr) in exprs.iter().enumerate() {
                        let expr_sql = fmt_sql(expr.as_ref());
                        writeln!(f, "expr[{i}][{j}]={expr_sql}")?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl ExecutionPlan for ExpandExec {
    fn name(&self) -> &str {
        "ExpandExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        ExpandExec::try_new_with_schema(
            self.exprs.clone(),
            children.swap_remove(0),
            Arc::clone(&self.schema),
        )
        .map(|p| Arc::new(p) as _)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion_execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(ExpandStream {
            schema: Arc::clone(&self.schema),
            exprs: self.exprs.clone(),
            input: self.input.execute(partition, context)?,
            baseline_metrics: BaselineMetrics::new(&self.metrics, partition),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(stats_expand(
            self.input.statistics()?,
            &self.exprs,
            Arc::clone(&self.schema),
        ))
    }
}

fn stats_expand(
    mut stats: Statistics,
    exprs: &[Vec<Arc<dyn PhysicalExpr>>],
    schema: SchemaRef,
) -> Statistics {
    let mut primitive_row_size = 0;
    let mut primitive_row_size_possible = true;
    let mut column_statistics = vec![];
    let column_num = schema.fields().len();
    for col_idx in 0..column_num {
        let mut col_stats: Option<ColumnStatistics> = None;
        for (row_idx, row) in exprs.iter().enumerate() {
            let expr = &row[col_idx];
            let cur_col_stats = if let Some(col) = expr.as_any().downcast_ref::<Column>()
            {
                stats.column_statistics[col.index()].clone()
            } else {
                ColumnStatistics::new_unknown()
            };
            match col_stats {
                Some(stats) => col_stats = Some(stats.merge(&cur_col_stats)),
                None => col_stats = Some(cur_col_stats),
            }

            if row_idx == 0 {
                if let Ok(data_type) = expr.data_type(&schema) {
                    if let Some(value) = data_type.primitive_width() {
                        primitive_row_size += value;
                        continue;
                    }
                }
            }
            primitive_row_size_possible = false;
        }
        column_statistics.push(col_stats.unwrap());
    }

    if primitive_row_size_possible {
        stats.total_byte_size =
            Precision::Exact(primitive_row_size).multiply(&stats.num_rows);
    }
    stats.column_statistics = column_statistics;
    stats
}

struct ExpandStream {
    schema: SchemaRef,
    exprs: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    input: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
}

impl ExpandStream {
    fn batch_expand(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        // Records time on drop
        let _timer = self.baseline_metrics.elapsed_compute().timer();

        let mut batches = Vec::new();
        for expr in self.exprs.iter() {
            let arrays = expr
                .iter()
                .map(|expr| {
                    expr.evaluate(batch)
                        .and_then(|v| v.into_array(batch.num_rows()))
                })
                .collect::<Result<Vec<_>>>()?;

            let batch = if arrays.is_empty() {
                let options =
                    RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                RecordBatch::try_new_with_options(
                    Arc::clone(&self.schema),
                    arrays,
                    &options,
                )
                .map_err(Into::<DataFusionError>::into)?
            } else {
                RecordBatch::try_new(Arc::clone(&self.schema), arrays)
                    .map_err(Into::<DataFusionError>::into)?
            };
            batches.push(batch);
        }
        arrow::compute::concat_batches(&self.schema, &batches)
            .map_err(Into::<DataFusionError>::into)
    }
}

impl Stream for ExpandStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let poll = self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(self.batch_expand(&batch)),
            other => other,
        });

        self.baseline_metrics.record_poll(poll)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for ExpandStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
