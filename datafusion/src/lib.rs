//! ArrowMetal inside Apache DataFusion: a physical optimizer rule and a custom `ExecutionPlan`.
//!
//! **Spike.** [`ArrowMetalRule`] walks DataFusion's optimized physical plan and replaces the nodes
//! ArrowMetal can run with the same answer — `SortExec` (with or without `fetch`), a hash
//! `AggregateExec` over column keys with `sum`/`min`/`max`/`count`/`avg`, and a `FilterExec` whose
//! predicate translates — by a [`MetalExec`]. `MetalExec` collects its input's partitions, runs the
//! operation through ArrowMetal's plan runner on the GPU, and emits `RecordBatch`es with the schema
//! DataFusion expects. Anything it does not support is left unchanged, and the reason is recorded in
//! the rule's [`Report`].
//!
//! ```no_run
//! # async fn f() -> datafusion::error::Result<()> {
//! use datafusion::prelude::*;
//! use datafusion_arrowmetal::{session_context, ArrowMetalConfig, ArrowMetalRule};
//!
//! let rule = ArrowMetalRule::new(ArrowMetalConfig { min_rows: 1_000_000, ..Default::default() });
//! let ctx = session_context(SessionConfig::new(), rule.clone());
//! // register tables, run SQL ...
//! for d in rule.report().decisions() { println!("{d}"); }
//! # Ok(()) }
//! ```

mod exec;
mod gpu;
mod rule;
mod translate;

pub use exec::{MetalExec, MetalOp};
pub use rule::{ArrowMetalConfig, ArrowMetalRule, Decision, Report};

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::prelude::{SessionConfig, SessionContext};

/// DataFusion's default physical optimizer rules with `rule` inserted before the last two:
/// the post-optimization `FilterPushdown` (which wires dynamic filters to the operators that own
/// them, so a node replaced after it would leave a dynamic filter nobody updates) and
/// `SanityCheckPlan` (so the rewritten plan is still checked for distribution and ordering).
pub fn physical_optimizer_rules(
    rule: ArrowMetalRule,
) -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    let mut rules = PhysicalOptimizer::new().rules;
    let at = rules
        .iter()
        .rposition(|r| r.name() == "SanityCheckPlan")
        .unwrap_or(rules.len());
    // The post-optimization FilterPushdown sits just before SanityCheckPlan in 55.1.0.
    let at = if at > 0 && rules[at - 1].name().starts_with("FilterPushdown") { at - 1 } else { at };
    rules.insert(at, Arc::new(rule));
    rules
}

/// Registers `rule` on a `SessionStateBuilder` (see [`physical_optimizer_rules`] for where).
pub fn with_arrowmetal(builder: SessionStateBuilder, rule: ArrowMetalRule) -> SessionStateBuilder {
    builder.with_physical_optimizer_rules(physical_optimizer_rules(rule))
}

/// A `SessionContext` with DataFusion's defaults plus `rule`.
pub fn session_context(config: SessionConfig, rule: ArrowMetalRule) -> SessionContext {
    let state = with_arrowmetal(
        SessionStateBuilder::new().with_config(config).with_default_features(),
        rule,
    )
    .build();
    SessionContext::new_with_state(state)
}
