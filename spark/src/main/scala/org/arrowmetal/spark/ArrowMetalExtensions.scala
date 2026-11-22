package org.arrowmetal.spark

import java.lang.foreign.{Arena, MemorySegment, ValueLayout}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.plans.physical.{Distribution, OrderedDistribution, Partitioning, UnspecifiedDistribution}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarRule, SortExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

/**
 * Spark on the Apple silicon GPU. Enable with
 *   --conf spark.sql.extensions=org.arrowmetal.spark.ArrowMetalExtensions
 * Settings: spark.arrowmetal.enabled (default true), spark.arrowmetal.sort.minRows (default 1000000:
 * smaller partitions are sorted by Spark's own ordering on the CPU), spark.arrowmetal.aggregate.enabled
 * (default true: the partial GROUP BY over a columnar scan, MetalPartialAggExec).
 */
class ArrowMetalExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(e: SparkSessionExtensions): Unit = e.injectColumnar(_ => new MetalColumnarRule)
}

class MetalColumnarRule extends ColumnarRule {
  override def preColumnarTransitions: Rule[SparkPlan] = MetalSortRule
  override def postColumnarTransitions: Rule[SparkPlan] = MetalAggRule
}

/** Replaces a SortExec on one key of a supported type with MetalSortExec. */
object MetalSortRule extends Rule[SparkPlan] {
  override def apply(plan: SparkPlan): SparkPlan = {
    if (!SQLConf.get.getConfString("spark.arrowmetal.enabled", "true").toBoolean) return plan
    plan.transformUp {
      case SortExec(order, global, child, _) if order.size == 1 && MetalSortExec.format(order.head.child.dataType).isDefined =>
        MetalSortExec(order, global, child)
    }
  }
}

case class MetalSortExec(sortOrder: Seq[SortOrder], global: Boolean, child: SparkPlan) extends UnaryExecNode {
  override def output: Seq[Attribute] = child.output
  override def outputOrdering: Seq[SortOrder] = sortOrder
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def requiredChildDistribution: Seq[Distribution] =
    if (global) OrderedDistribution(sortOrder) :: Nil else UnspecifiedDistribution :: Nil
  override protected def withNewChildInternal(newChild: SparkPlan): MetalSortExec = copy(child = newChild)

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "gpuRows" -> SQLMetrics.createMetric(sparkContext, "rows sorted on the GPU"),
    "cpuRows" -> SQLMetrics.createMetric(sparkContext, "rows sorted on the CPU"),
    "gpuTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "GPU sort time, incl. key copy"))

  override protected def doExecute(): RDD[InternalRow] = {
    val order = sortOrder.head
    val key = BindReferences.bindReference(order.child, child.output)
    val (fmt, width) = MetalSortExec.format(order.child.dataType).get
    val descending = order.direction == Descending
    val nullsFirst = order.nullOrdering == NullsFirst
    val minRows = conf.getConfString("spark.arrowmetal.sort.minRows", "1000000").toLong
    val cpuOrdering = new LazilyGeneratedOrdering(sortOrder, child.output)
    val gpuRows = longMetric("gpuRows")
    val cpuRows = longMetric("cpuRows")
    val gpuTime = longMetric("gpuTime")
    child.execute().mapPartitions({ iter =>
      val rows = new ArrayBuffer[InternalRow]
      while (iter.hasNext) rows += iter.next().copy()
      val n = rows.length
      if (n < minRows) {
        cpuRows += n
        rows.sortWith((a, b) => cpuOrdering.compare(a, b) < 0).iterator
      } else {
        val t0 = System.nanoTime()
        val idx = MetalSortExec.argsort(rows, key, fmt, width, descending, nullsFirst)
        gpuTime += System.nanoTime() - t0
        gpuRows += n
        idx.iterator.map(i => rows(i))
      }
    }, preservesPartitioning = true)
  }
}

object MetalSortExec {
  /** Arrow format string and width for a Spark key type whose order is its storage order. */
  def format(dt: DataType): Option[(String, Int)] = dt match {
    case LongType | TimestampType | TimestampNTZType => Some(("l", 8))
    case IntegerType | DateType => Some(("i", 4))
    case DoubleType => Some(("g", 8))
    case FloatType => Some(("f", 4))
    case _ => None
  }

  def argsort(rows: ArrayBuffer[InternalRow], key: Expression, fmt: String, width: Int,
              descending: Boolean, nullsFirst: Boolean): Array[Int] = {
    val n = rows.length
    val arena = Arena.ofConfined()
    try {
      val values = arena.allocate(width.toLong * n, 64)
      val validity = arena.allocate(((n + 63) / 64) * 8L, 64)     // zeroed: every row starts null
      var nulls = 0L
      val ordinal = key match { case b: BoundReference => b.ordinal; case _ => -1 }
      var i = 0
      while (i < n) {
        val r = rows(i)
        val isNull = if (ordinal >= 0) r.isNullAt(ordinal) else key.eval(r) == null
        if (isNull) nulls += 1
        else {
          val byte = i >>> 3
          validity.set(ValueLayout.JAVA_BYTE, byte.toLong, (validity.get(ValueLayout.JAVA_BYTE, byte.toLong) | (1 << (i & 7))).toByte)
          fmt match {
            case "l" => values.setAtIndex(ValueLayout.JAVA_LONG, i.toLong, if (ordinal >= 0) r.getLong(ordinal) else key.eval(r).asInstanceOf[Long])
            case "i" => values.setAtIndex(ValueLayout.JAVA_INT, i.toLong, if (ordinal >= 0) r.getInt(ordinal) else key.eval(r).asInstanceOf[Int])
            case "g" => values.setAtIndex(ValueLayout.JAVA_DOUBLE, i.toLong, if (ordinal >= 0) r.getDouble(ordinal) else key.eval(r).asInstanceOf[Double])
            case "f" => values.setAtIndex(ValueLayout.JAVA_FLOAT, i.toLong, if (ordinal >= 0) r.getFloat(ordinal) else key.eval(r).asInstanceOf[Float])
          }
        }
        i += 1
      }
      ArrowMetalNative.argsort(values, validity, n.toLong, nulls, fmt, descending, nullsFirst)
    } finally arena.close()
  }
}
