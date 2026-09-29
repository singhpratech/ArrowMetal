//! ArrowMetal under DataFusion: one table, three queries, the rule's report and EXPLAIN for each.
//!
//!     ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --example quickstart

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::prelude::SessionConfig;
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};

#[tokio::main]
async fn main() -> Result<()> {
    let n = 200_000usize;
    let region: Int64Array = (0..n).map(|i| Some((i * 7919 % 13) as i64)).collect();
    let amount: Float64Array =
        (0..n).map(|i| if i % 97 == 0 { None } else { Some(((i * 104_729) % 10_007) as f64 / 7.0) }).collect();
    let name: StringArray = (0..n).map(|i| Some(format!("n{:05}", (i * 31) % 50_000))).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Int64, false),
        Field::new("amount", DataType::Float64, true),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(region), Arc::new(amount), Arc::new(name)],
    )?;

    // A small table, so the threshold is lowered from its 250,000-row default.
    let rule = ArrowMetalRule::new(ArrowMetalConfig::all().with_min_rows(100_000));
    let ctx = session_context(SessionConfig::new().with_target_partitions(4), rule.clone());
    ctx.register_table("sales", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;

    for sql in [
        "SELECT region, sum(amount) AS total, count(*) AS n, avg(amount) AS mean, max(amount) AS top \
         FROM sales GROUP BY region ORDER BY total DESC",
        "SELECT name, amount FROM sales ORDER BY amount DESC NULLS LAST, name LIMIT 5",
        "SELECT region, amount FROM sales WHERE region = 3 AND amount > 1000.5 ORDER BY amount LIMIT 3",
    ] {
        rule.clear_report();
        println!("== {sql}\n");
        let df = ctx.sql(sql).await?;
        let plan = df.clone().create_physical_plan().await?;
        println!(
            "{}",
            datafusion::physical_plan::displayable(plan.as_ref()).indent(false)
        );
        println!("report:\n{}", rule.report());
        let out = df.collect().await?;
        println!("{}\n", pretty_format_batches(&out)?);
    }
    Ok(())
}
