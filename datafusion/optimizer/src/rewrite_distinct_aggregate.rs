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

use std::{collections::HashMap, sync::Arc};

use arrow::datatypes::{DataType, Field};
use datafusion_common::{
    internal_err, tree_node::Transformed, HashSet, Result, ScalarValue,
};
use datafusion_expr::builder::project;
use datafusion_expr::logical_plan::Expand;
use datafusion_expr::sqlparser::ast::NullTreatment;
use datafusion_expr::SortExpr;
use datafusion_expr::{
    col, expr::AggregateFunction, lit, Aggregate, Expr, ExprSchemable,
    LogicalPlan,
};
use datafusion_functions_aggregate::first_last::first_value_udaf;
use datafusion_functions_aggregate::min_max::max;
use itertools::Itertools;

use crate::{ApplyOrder, OptimizerConfig, OptimizerRule};

#[derive(Default, Debug)]
pub struct RewriteDistinctAggregate {}

impl RewriteDistinctAggregate {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

/**
 * This rule rewrites an aggregate query with distinct aggregations into an expanded double
 * aggregation in which the regular aggregation expressions and every distinct clause is aggregated
 * in a separate group. The results are then combined in a second aggregate.
 *
 * First example: query without filter clauses:
 * ```
 *   let data = vec![
 *     ("a", "ca1", "cb1", 10),
 *     ("a", "ca1", "cb2", 5),
 *     ("b", "ca1", "cb1", 13)
 *   ];
 *   // Create DataFrame from data
 *
 *   // Group by key and aggregate with distinct counts and sum
 *   let agg = df.group_by(vec![col("key")])?
 *     .aggregate(vec![
 *       count_distinct(col("cat1")).alias("cat1_cnt"),
 *       count_distinct(col("cat2")).alias("cat2_cnt"),
 *       sum(col("value")).alias("total")
 *     ])?;
 * ```
 *
 * This translates to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(DISTINCT cat1), COUNT(DISTINCT cat2), SUM(value)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--TableScan: data projection=[key, cat1, cat2, value]
 * ```
 *
 * This rule rewrites this logical plan to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(cat1) FILTER (WHERE gid = 1), COUNT(cat2) FILTER (WHERE gid = 2), FIRST_VALUE(total) FILTER (WHERE gid = 0)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--Aggregate: groupBy=[[key, cat1, cat2, gid]], aggr=[[SUM(value)]], output=[[key, cat1, cat2, gid, total]]
 * 03)----Unnest: projections=[[(key, NULL, NULL, 0, CAST(value AS bigint)), (key, cat1, NULL, 1, NULL), (key, NULL, cat2, 2, NULL)]], output=[[key, cat1, cat2, gid, value]]
 * 04)------TableScan: data projection=[key, cat1, cat2, value]
 * ```
 *
 * Second example: aggregate function without distinct and with filter clauses:
 * ```sql
 *   SELECT
 *     COUNT(DISTINCT cat1) as cat1_cnt,
 *     COUNT(DISTINCT cat2) as cat2_cnt,
 *     SUM(value) FILTER (WHERE id > 1) AS total
 *   FROM
 *     data
 *   GROUP BY
 *     key
 * ```
 *
 * This translates to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(DISTINCT cat1), COUNT(DISTINCT cat2), SUM(value) FILTER (WHERE id > 1)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--TableScan: data projection=[key, cat1, cat2, value, id]
 * ```
 *
 * This rule rewrites this logical plan to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(cat1) FILTER (WHERE gid = 1), COUNT(cat2) FILTER (WHERE gid = 2), FIRST_VALUE(total) FILTER (WHERE gid = 0)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--Aggregate: groupBy=[[key, cat1, cat2, gid]], aggr=[[SUM(value) FILTER (WHERE id > 1)]], output=[[key, cat1, cat2, gid, total]]
 * 03)----Unnest: projections=[[(key, NULL, NULL, 0, CAST(value AS bigint), id), (key, cat1, NULL, 1, NULL, NULL), (key, NULL, cat2, 2, NULL, NULL)]], output=[[key, cat1, cat2, gid, value, id]]
 * 04)------TableScan: data projection=[key, cat1, cat2, value, id]
 * ```
 *
 * Third example: aggregate function with distinct and filter clauses:
 * ```sql
 *   SELECT
 *     COUNT(DISTINCT cat1) FILTER (WHERE id > 1) as cat1_cnt,
 *     COUNT(DISTINCT cat2) FILTER (WHERE id > 2) as cat2_cnt,
 *     SUM(value) FILTER (WHERE id > 3) AS total
 *   FROM
 *     data
 *   GROUP BY
 *     key
 * ```
 *
 * This translates to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(DISTINCT cat1) FILTER (WHERE id > 1), COUNT(DISTINCT cat2) FILTER (WHERE id > 2), SUM(value) FILTER (WHERE id > 3)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--TableScan: data projection=[key, cat1, cat2, value, id]
 * ```
 *
 * This rule rewrites this logical plan to the following logical plan:
 * ```
 * 01)Aggregate: groupBy=[[key]], aggr=[[COUNT(cat1) FILTER (WHERE gid = 1 AND max_cond1), COUNT(cat2) FILTER (WHERE gid = 2 AND max_cond2), FIRST_VALUE(total) FILTER (WHERE gid = 0)]], output=[[key, cat1_cnt, cat2_cnt, total]]
 * 02)--Aggregate: groupBy=[[key, cat1, cat2, gid]], aggr=[[MAX(cond1), MAX(cond2), SUM(value) FILTER (WHERE id > 3)]], output=[[key, cat1, cat2, gid, max_cond1, max_cond2, total]]
 * 03)----Unnest: projections=[[(key, NULL, NULL, 0, NULL, NULL, CAST(value AS bigint), id), (key, cat1, NULL, 1, id > 1, NULL, NULL, NULL), (key, NULL, cat2, 2, NULL, id > 2, NULL, NULL)]], output=[[key, cat1, cat2, gid, cond1, cond2, value, id]]
 * 04)------TableScan: data projection=[key, cat1, cat2, value, id]
 * ```
 *
 * The rule does the following things here:
 * 1. Expand the data. There are three aggregation groups in this query:
 *    i. the non-distinct group;
 *    ii. the distinct 'cat1 group;
 *    iii. the distinct 'cat2 group.
 *    An expand operator is inserted to expand the child data for each group. The expand will null
 *    out all unused columns for the given group; this must be done in order to ensure correctness
 *    later on. Groups can by identified by a group id (gid) column added by the expand operator.
 *    If distinct group exists filter clause, the expand will calculate the filter and output it's
 *    result (e.g. cond1) which will be used to calculate the global conditions (e.g. max_cond1)
 *    equivalent to filter clauses.
 * 2. De-duplicate the distinct paths and aggregate the non-aggregate path. The group by clause of
 *    this aggregate consists of the original group by clause, all the requested distinct columns
 *    and the group id. Both de-duplication of distinct column and the aggregation of the
 *    non-distinct group take advantage of the fact that we group by the group id (gid) and that we
 *    have nulled out all non-relevant columns the given group. If distinct group exists filter
 *    clause, we will use max to aggregate the results (e.g. cond1) of the filter output in the
 *    previous step. These aggregate will output the global conditions (e.g. max_cond1) equivalent
 *    to filter clauses.
 * 3. Aggregating the distinct groups and combining this with the results of the non-distinct
 *    aggregation. In this step we use the group id and the global condition to filter the inputs
 *    for the aggregate functions. If the global condition (e.g. max_cond1) is true, it means at
 *    least one row of a distinct value satisfies the filter. This distinct value should be included
 *    in the aggregate function. The result of the non-distinct group are 'aggregated' by using
 *    the first operator, it might be more elegant to use the native UDAF merge mechanism for this
 *    in the future.
 *
 * This rule duplicates the input data by two or more times (# distinct groups + an optional
 * non-distinct group). This will put quite a bit of memory pressure of the used aggregate and
 * exchange operators. Keeping the number of distinct groups as low as possible should be priority,
 * we could improve this in the current rule by applying more advanced expression canonicalization
 * techniques.
 */
impl OptimizerRule for RewriteDistinctAggregate {
    fn name(&self) -> &str {
        "rewrite_distinct_aggregate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Aggregate(Aggregate {
                ref input,
                ref aggr_expr,
                ref schema,
                ref group_expr,
                ..
            }) => {
                let aggr_expr = aggr_expr
                    .iter()
                    .map(|expr| match expr {
                        Expr::AggregateFunction(func) => Ok(func),
                        _ => internal_err!("Expected aggregate function"),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (distinct_exprs, regular_exprs): (
                    Vec<&AggregateFunction>,
                    Vec<&AggregateFunction>,
                ) = aggr_expr.iter().partition(|func| func.params.distinct);

                let distinct_groups = distinct_exprs
                    .iter()
                    .map(|f| distinct_group(f).map(|e| (e, f)))
                    .collect::<Result<Vec<_>>>()?;
                let distinct_groups = distinct_groups
                    .iter()
                    .chunk_by(|(args, _)| args)
                    .into_iter()
                    .map(|(key, group)| {
                        (key, group.map(|(_, expr)| *expr).collect::<Vec<_>>())
                    })
                    .collect::<Vec<_>>();

                if distinct_groups.len() <= 1
                    && !must_rewrite(group_expr, &distinct_exprs)
                {
                    return Ok(Transformed::no(plan));
                }

                let distinct_expr_list = distinct_groups
                    .iter()
                    .map(|(key, _)| **key)
                    .filter(|expr| !is_constant(expr))
                    .dedup()
                    .collect::<Vec<_>>();

                let regular_expr_list = regular_exprs
                    .iter()
                    .flat_map(|f| f.params.args.iter())
                    .filter(|expr| !is_constant(expr))
                    .dedup()
                    .collect::<Vec<_>>();
                let filter_expr_list = aggr_expr
                    .iter()
                    .flat_map(|expr| expr.params.filter.iter())
                    .filter(|expr| !is_constant(expr))
                    .dedup()
                    .collect::<Vec<_>>();

                let order_by_expr_list = aggr_expr
                    .iter()
                    .flat_map(|expr| expr.params.order_by.iter())
                    .flat_map(|expr| expr.iter())
                    .map(|expr| &expr.expr)
                    .filter(|expr| !is_constant(expr))
                    .dedup()
                    .collect::<Vec<_>>();

                let group_expr_idx_map = group_expr
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (expr, i))
                    .collect::<HashMap<_, _>>();

                let distinct_expr_idx_map = distinct_expr_list
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (*expr, i))
                    .collect::<HashMap<_, _>>();

                let regular_expr_idx_map = regular_expr_list
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (*expr, i))
                    .collect::<HashMap<_, _>>();

                let filter_expr_idx_map = filter_expr_list
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (*expr, i))
                    .collect::<HashMap<_, _>>();

                let order_by_expr_idx_map = order_by_expr_list
                    .iter()
                    .enumerate()
                    .map(|(i, expr)| (*expr, i))
                    .collect::<HashMap<_, _>>();

                // group_exprs + distinct_exprs + gid + filter_exprs + regular_exprs + order_by_exprs
                let mut exprs_in_array = vec![];

                if !regular_expr_list.is_empty() {
                    let regular_filter_expr_set = regular_exprs
                        .iter()
                        .flat_map(|f| f.params.filter.iter())
                        .filter(|expr| !is_constant(expr))
                        .dedup()
                        .collect::<HashSet<_>>();
                    let mut exprs = vec![];
                    // group_exprs
                    exprs.extend(group_expr.clone());
                    // distinct_exprs
                    for _ in distinct_expr_list.iter() {
                        exprs.push(Expr::Literal(ScalarValue::Null));
                    }
                    // gid
                    exprs.push(Expr::Literal(ScalarValue::Int64(Some(0))));
                    // filter_exprs
                    for expr in filter_expr_list.iter() {
                        if regular_filter_expr_set.contains(expr) {
                            exprs.push(expr.as_ref().clone());
                        } else {
                            exprs.push(Expr::Literal(ScalarValue::Null));
                        }
                    }
                    // regular_exprs
                    for expr in regular_expr_list.iter() {
                        exprs.push((*expr).clone());
                    }
                    // order_by_exprs
                    for expr in order_by_expr_list.iter() {
                        exprs.push((*expr).clone());
                    }
                    exprs_in_array.push(exprs);
                }

                for (idx, (key, distinct_aggs)) in distinct_groups.iter().enumerate() {
                    let mut exprs = vec![];
                    let distinct_filter_expr_set = distinct_aggs
                        .iter()
                        .flat_map(|f| f.params.filter.iter())
                        .filter(|expr| !is_constant(expr))
                        .dedup()
                        .collect::<HashSet<_>>();
                    // group_exprs
                    exprs.extend(group_expr.clone());
                    // distinct_exprs
                    for expr in distinct_expr_list.iter() {
                        if key == &expr {
                            exprs.push((*expr).clone());
                        } else {
                            exprs.push(Expr::Literal(ScalarValue::Null));
                        }
                    }
                    // gid
                    exprs.push(Expr::Literal(ScalarValue::Int64(Some(idx as i64 + 1))));
                    // filter_exprs
                    for expr in filter_expr_list.iter() {
                        if distinct_filter_expr_set.contains(expr) {
                            exprs.push(expr.as_ref().clone());
                        } else {
                            exprs.push(Expr::Literal(ScalarValue::Null));
                        }
                    }
                    // regular_exprs
                    for _ in regular_expr_list.iter() {
                        exprs.push(Expr::Literal(ScalarValue::Null));
                    }
                    // order_by_exprs
                    for _ in order_by_expr_list.iter() {
                        exprs.push(Expr::Literal(ScalarValue::Null));
                    }
                    exprs_in_array.push(exprs);
                }

                let mut fields = vec![];
                for (expr, idx) in group_expr_idx_map.iter() {
                    let (data_type, nullable) = expr.data_type_and_nullable(schema)?;
                    fields.push(Field::new(
                        format!("group_{}", idx),
                        data_type,
                        nullable,
                    ));
                }

                for (expr, idx) in distinct_expr_idx_map.iter() {
                    let (data_type, nullable) = expr.data_type_and_nullable(schema)?;
                    fields.push(Field::new(format!("cat_{}", idx), data_type, nullable));
                }

                fields.push(Field::new("gid".to_string(), DataType::Int64, false));

                for (expr, idx) in filter_expr_idx_map.iter() {
                    let (data_type, nullable) = expr.data_type_and_nullable(schema)?;
                    fields.push(Field::new(
                        format!("filter_{}", idx),
                        data_type,
                        nullable,
                    ));
                }

                for (expr, idx) in regular_expr_idx_map.iter() {
                    let (data_type, nullable) = expr.data_type_and_nullable(schema)?;
                    fields.push(Field::new(
                        format!("regular_{}", idx),
                        data_type,
                        nullable,
                    ));
                }

                let expand =
                    LogicalPlan::Expand(Expand::try_new(exprs_in_array, Arc::clone(input))?);

                let init_group_expr = group_expr
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| col(format!("group_{}", idx)))
                    .chain(
                        distinct_expr_list
                            .iter()
                            .enumerate()
                            .map(|(idx, _)| col(format!("cat_{}", idx))),
                    )
                    .chain(std::iter::once(col("gid")))
                    .collect();

                let init_agg_expr = filter_expr_list
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| max(col(format!("filter_{}", idx))))
                    .chain(regular_exprs.iter().map(|f| {
                        let args = f
                            .params
                            .args
                            .iter()
                            .map(|e| {
                                regular_expr_idx_map
                                    .get(e)
                                    .map(|idx| col(format!("regular_{}", idx)))
                                    .unwrap_or_else(|| e.clone())
                            })
                            .collect();

                        let filter = f.params.filter.as_ref().map(|e| {
                            Box::new(
                                regular_expr_idx_map
                                    .get(e.as_ref())
                                    .map(|idx| col(format!("filter_{}", idx)))
                                    .unwrap_or_else(|| e.as_ref().clone()),
                            )
                        });

                        let order_by: Option<Vec<SortExpr>> =
                            f.params.order_by.as_ref().map(|order_by| {
                                order_by
                                    .iter()
                                    .map(|s| {
                                        SortExpr::new(
                                            order_by_expr_idx_map
                                                .get(&s.expr)
                                                .map(|idx| {
                                                    col(format!("order_by_{}", idx))
                                                })
                                                .unwrap_or_else(|| s.expr.clone()),
                                            s.asc,
                                            s.nulls_first,
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            });
                        Expr::AggregateFunction(AggregateFunction::new_udf(
                            Arc::clone(&f.func),
                            args,
                            false,
                            filter,
                            order_by,
                            f.params.null_treatment,
                        ))
                    }))
                    .collect();

                let init_agg = LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::new(expand),
                    init_group_expr,
                    init_agg_expr,
                )?);

                let final_group_expr = group_expr
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| col(format!("group_{}", idx)))
                    .collect::<Vec<_>>();
                let mut alias_expr = vec![];
                let mut final_agg_expr = vec![];

                for (idx, f) in aggr_expr.iter().enumerate() {
                    let new_f = if f.params.distinct {
                        let group = distinct_group(f)?;
                        let gid = (*distinct_expr_idx_map.get(group).unwrap()) as i64;
                        let mut filter = col("gid").eq(lit(gid));
                        if let Some(e) = f.params.filter.as_ref() {
                            filter = filter.and(
                                regular_expr_idx_map
                                    .get(e.as_ref())
                                    .map(|idx| col(format!("filter_{}", idx)))
                                    .unwrap_or_else(|| e.as_ref().clone()),
                            )
                        }
                        let order_by: Option<Vec<SortExpr>> =
                            f.params.order_by.as_ref().map(|order_by| {
                                order_by
                                    .iter()
                                    .map(|s| {
                                        SortExpr::new(
                                            order_by_expr_idx_map
                                                .get(&s.expr)
                                                .map(|idx| {
                                                    col(format!("order_by_{}", idx))
                                                })
                                                .unwrap_or_else(|| s.expr.clone()),
                                            s.asc,
                                            s.nulls_first,
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            });

                        AggregateFunction::new_udf(
                            Arc::clone(&f.func),
                            vec![group.clone()],
                            false,
                            Some(Box::new(filter)),
                            order_by,
                            f.params.null_treatment,
                        )
                    } else {
                        let args = f
                            .params
                            .args
                            .iter()
                            .map(|e| {
                                regular_expr_idx_map
                                    .get(e)
                                    .map(|idx| col(format!("regular_{}", idx)))
                                    .unwrap_or_else(|| e.clone())
                            })
                            .collect();
                        AggregateFunction::new_udf(
                            first_value_udaf(),
                            args,
                            false,
                            Some(Box::new(lit("gid").eq(lit(0)))),
                            None,
                            Some(NullTreatment::IgnoreNulls),
                        )
                    };
                    let new_expr = Expr::AggregateFunction(new_f);

                    let (qualifier, field) = schema.qualified_field(idx);
                    alias_expr.push(
                        new_expr
                            .clone()
                            .alias_qualified(qualifier.cloned(), field.name()),
                    );
                    final_agg_expr.push(new_expr);
                }

                let final_agg = LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::new(init_agg),
                    final_group_expr,
                    final_agg_expr,
                )?);
                Ok(Transformed::yes(project(final_agg, alias_expr)?))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

fn must_rewrite(group_expr: &[Expr], distinct_exprs: &[&AggregateFunction]) -> bool {
    // If there are any distinct AggregateExpressions with filter, we need to rewrite the query.
    // Also, if there are no grouping expressions and all distinct aggregate expressions are
    // foldable, we need to rewrite the query, e.g. SELECT COUNT(DISTINCT 1). Without this case,
    // non-grouping aggregation queries with distinct aggregate expressions will be incorrectly
    // handled by the aggregation strategy, causing wrong results when working with empty tables.
    if distinct_exprs.iter().any(|f| f.params.filter.is_some()) {
        return true;
    }

    group_expr.is_empty()
        && distinct_exprs
            .iter()
            .all(|f| f.params.args.iter().all(is_constant))
}

fn is_constant(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(_))
}

pub fn distinct_group(f: &AggregateFunction) -> Result<&Expr> {
    let args = &f.params.args;
    if args.len() != 1 {
        return internal_err!("DISTINCT aggregate should have exactly one argument");
    }
    Ok(&args[0])
}
