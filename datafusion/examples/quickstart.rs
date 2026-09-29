//! ArrowMetal under DataFusion with the default configuration: a full `ORDER BY` runs on the GPU;
//! a top-k and a `GROUP BY` stay on DataFusion. For each query it prints the physical plan, the
//! rule's report and the first rows of the answer. This is the example in docs/DATAFUSION.md.
//!
//! ```text
//! ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib cargo run --example quickstart
//! ```

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionConfig;
use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};

#[tokio::main]
async fn main() -> Result<()> {
    // One million rows: above the default threshold (`min_rows`, 250,000).
    let n = 1_000_000usize;
    let region: Int64Array = (0..n).map(|i| Some((i * 7919 % 13) as i64)).collect();
    let amount: Float64Array = (0..n)
        .map(|i| if i % 97 == 0 { None } else { Some(((i * 104_729) % 10_007) as f64 / 7.0) })
        .collect();
    let name: StringArray = (0..n).map(|i| Some(format!("n{:06}", (i * 31) % 500_000))).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Int64, false),
        Field::new("amount", DataType::Float64, true),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(region), Arc::new(amount), Arc::new(name)],
    )?;

    // The default configuration: full sorts from 250,000 rows. Four partitions keep the printed
    // plans short; DataFusion's own default is one per core.
    let rule = ArrowMetalRule::new(ArrowMetalConfig::default());
    let ctx = session_context(SessionConfig::new().with_target_partitions(4), rule.clone());
    ctx.register_table("sales", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;

    for sql in [
        "SELECT name, region, amount FROM sales ORDER BY amount DESC NULLS LAST, name",
        "SELECT name, amount FROM sales ORDER BY amount DESC NULLS LAST LIMIT 3",
        "SELECT region, count(*) AS n, avg(amount) AS mean FROM sales GROUP BY region ORDER BY region",
    ] {
        rule.clear_report();
        println!("== {sql}\n");
        // Plan once and run that plan, so the report holds one planning pass.
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        println!("{}", displayable(plan.as_ref()).indent(false));
        println!("{}", rule.report());
        let out = collect(plan, ctx.task_ctx()).await?;
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        let head = out.first().map(|b| b.slice(0, b.num_rows().min(3)));
        println!("{rows} rows; the first {}:", head.as_ref().map_or(0, |b| b.num_rows()));
        println!("{}\n", pretty_format_batches(head.as_slice())?);
    }
    Ok(())
}
