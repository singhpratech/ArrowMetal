package org.arrowmetal.spark

import java.lang.foreign.{Arena, MemorySegment, ValueLayout}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarToRowExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * The partial (map-side) half of a GROUP BY, on the GPU, fed by the column batches of a columnar scan
 * (Spark's vectorized Parquet or ORC reader) with no row conversion. One key of int, long or date;
 * aggregates SUM(int), SUM(double), COUNT(*), COUNT(col), MIN/MAX(int or long). Anything else, or a
 * child that is not columnar, stays on Spark's HashAggregateExec. SUM of longs stays there too: Spark
 * raises on overflow in ANSI mode and the GPU sum wraps.
 */
object MetalAggRule extends Rule[SparkPlan] {
  override def apply(plan: SparkPlan): SparkPlan = {
    if (!SQLConf.get.getConfString("spark.arrowmetal.enabled", "true").toBoolean ||
        !SQLConf.get.getConfString("spark.arrowmetal.aggregate.enabled", "true").toBoolean) return plan
    // Runs after Spark has put ColumnarToRowExec in front of every row-based consumer of a columnar
    // scan: the GPU operator takes the scan's batches itself, so the converter comes out.
    plan.transformUp {
      case agg @ HashAggregateExec(_, _, _, _, _, _, _, _, ColumnarToRowExec(scan))
          if MetalPartialAggExec.plan(agg, scan).isDefined => MetalPartialAggExec(agg, scan)
    }
  }
}

/** What the GPU computes for one aggregate: an op of am_group_agg_ex on an input column (-1: none). */
case class MetalAgg(op: Int, input: Int, inputType: DataType, resultType: DataType)

case class MetalPartialAggExec(agg: HashAggregateExec, child: SparkPlan) extends UnaryExecNode {
  override def output: Seq[Attribute] = agg.output
  override protected def withNewChildInternal(newChild: SparkPlan): MetalPartialAggExec = copy(child = newChild)
  override def nodeName: String = "MetalPartialAggregate"

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "inputRows" -> SQLMetrics.createMetric(sparkContext, "rows aggregated on the GPU"),
    "groups" -> SQLMetrics.createMetric(sparkContext, "groups out"),
    "copyTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "column copy time"),
    "gpuTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "GPU group-by time"))

  override protected def doExecute(): RDD[InternalRow] = {
    val (keyIdx, keyType, aggs) = MetalPartialAggExec.plan(agg, child).get
    val outTypes = output.map(_.dataType)
    val m = metrics
    child.executeColumnar().mapPartitions { batches =>
      MetalPartialAggExec.run(batches, keyIdx, keyType, aggs, outTypes, m)
    }
  }
}

object MetalPartialAggExec {
  private def width(dt: DataType): Int = dt match {
    case IntegerType | DateType => 4
    case LongType | DoubleType => 8
  }
  private def format(dt: DataType): String = dt match {
    case IntegerType | DateType => "i"
    case LongType => "l"
    case DoubleType => "g"
  }

  /** (key column, key type, aggregates) when the GPU can compute `agg` exactly as Spark would. */
  def plan(agg: HashAggregateExec, scan: SparkPlan): Option[(Int, DataType, Seq[MetalAgg])] = {
    if (!scan.supportsColumnar) return None
    if (agg.aggregateExpressions.isEmpty || !agg.aggregateExpressions.forall(e => e.mode == Partial && !e.isDistinct && e.filter.isEmpty)) return None
    if (agg.groupingExpressions.size != 1) return None
    val in = scan.output
    def col(e: Expression): Option[Int] = e match {
      case a: AttributeReference => Some(in.indexWhere(_.exprId == a.exprId)).filter(_ >= 0)
      case _ => None
    }
    val key = agg.groupingExpressions.head
    val keyIdx = col(key).getOrElse(return None)
    val keyType = key.dataType
    if (!Seq(IntegerType, LongType, DateType).contains(keyType)) return None
    val aggs = agg.aggregateExpressions.map(_.aggregateFunction).map {
      case f if f.aggBufferAttributes.size != 1 => return None
      case s: Sum if s.child.dataType == IntegerType || s.child.dataType == DoubleType =>
        MetalAgg(0, col(s.child).getOrElse(return None), s.child.dataType, s.dataType)
      case c: Count if c.children.size == 1 && c.children.head.foldable && c.children.head.eval() != null =>
        MetalAgg(1, -1, NullType, LongType)
      case c: Count if c.children.size == 1 && Seq(IntegerType, LongType, DoubleType, DateType).contains(c.children.head.dataType) =>
        MetalAgg(2, col(c.children.head).getOrElse(return None), c.children.head.dataType, LongType)
      case f: Min if f.child.dataType == IntegerType || f.child.dataType == LongType =>
        MetalAgg(4, col(f.child).getOrElse(return None), f.child.dataType, f.dataType)
      case f: Max if f.child.dataType == IntegerType || f.child.dataType == LongType =>
        MetalAgg(5, col(f.child).getOrElse(return None), f.child.dataType, f.dataType)
      case _ => return None
    }
    if (agg.output.size != 1 + aggs.size) return None
    Some((keyIdx, keyType, aggs))
  }

  /** One column gathered from a partition's batches into off-heap memory, with an Arrow validity bitmap. */
  private final class Gather(dt: DataType) {
    val w: Int = width(dt)
    val chunks = new ArrayBuffer[AnyRef]
    val nullRows = new ArrayBuffer[Long]
    var n = 0L
    def add(b: ColumnarBatch, idx: Int): Unit = {
      val v = b.column(idx)
      val rows = b.numRows()
      chunks += (dt match {
        case IntegerType | DateType => v.getInts(0, rows)
        case LongType => v.getLongs(0, rows)
        case DoubleType => v.getDoubles(0, rows)
      })
      if (v.hasNull) { var i = 0; while (i < rows) { if (v.isNullAt(i)) nullRows += n + i; i += 1 } }
      n += rows
    }
    def importInto(arena: Arena): MemorySegment = {
      val values = arena.allocate(math.max(1L, w * n), 64)
      var at = 0L
      chunks.foreach {
        case a: Array[Int] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_INT, at, a.length); at += 4L * a.length
        case a: Array[Long] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_LONG, at, a.length); at += 8L * a.length
        case a: Array[Double] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_DOUBLE, at, a.length); at += 8L * a.length
      }
      chunks.clear()
      var validity = MemorySegment.NULL
      if (nullRows.nonEmpty) {
        val bytes = ((n + 63) / 64) * 8
        validity = arena.allocate(bytes, 64)
        validity.fill(0xFF.toByte)
        nullRows.foreach { r =>
          val b = r >>> 3
          validity.set(ValueLayout.JAVA_BYTE, b, (validity.get(ValueLayout.JAVA_BYTE, b) & ~(1 << (r & 7)).toInt).toByte)
        }
      }
      ArrowMetalNative.importColumn(arena, format(dt), values, validity, n, nullRows.size.toLong)
    }
  }

  def run(batches: Iterator[ColumnarBatch], keyIdx: Int, keyType: DataType, aggs: Seq[MetalAgg],
          outTypes: Seq[DataType], m: Map[String, SQLMetric]): Iterator[InternalRow] = {
    val t0 = System.nanoTime()
    val key = new Gather(keyType)
    val inputs = aggs.map(_.input).filter(_ >= 0).distinct
    val gathers = inputs.map(i => i -> new Gather(aggs.find(_.input == i).get.inputType)).toMap
    while (batches.hasNext) {
      val b = batches.next()
      key.add(b, keyIdx)
      gathers.foreach { case (i, g) => g.add(b, i) }
    }
    if (key.n == 0) return Iterator.empty
    val arena = Arena.ofConfined()
    val t1 = System.nanoTime()
    var keys: ArrowMetalNative.Column = null
    var results: Seq[ArrowMetalNative.Column] = null
    var t2 = 0L
    try {
      val keyArr = key.importInto(arena)
      val inArr = gathers.map { case (i, g) => i -> g.importInto(arena) }
      t2 = System.nanoTime()
      val gb = ArrowMetalNative.groupBy(keyArr)
      try {
        keys = ArrowMetalNative.groupKeys(gb)
        results = aggs.map(a => ArrowMetalNative.groupAgg(gb, if (a.input >= 0) inArr(a.input) else null, a.op))
      } finally {
        ArrowMetalNative.groupRelease(gb)
        ArrowMetalNative.release(keyArr)
        inArr.values.foreach(ArrowMetalNative.release)
      }
    } finally arena.close()
    val t3 = System.nanoTime()
    m("copyTime") += (t1 - t0) + (t2 - t1)
    m("gpuTime") += t3 - t2
    m("inputRows") += key.n
    m("groups") += keys.length
    val proj = UnsafeProjection.create(outTypes.toArray)
    val row = new SpecificInternalRow(outTypes)
    Iterator.range(0, keys.length).map { g =>
      if (keys.isNull(g)) row.setNullAt(0)
      else keyType match {
        case LongType => row.setLong(0, keys.longs(g))
        case _ => row.setInt(0, keys.ints(g))
      }
      var j = 0
      while (j < aggs.size) {
        val a = aggs(j)
        val c = results(j)
        if (c.isNull(g)) {
          if (a.op == 1 || a.op == 2) row.setLong(j + 1, 0L) else row.setNullAt(j + 1)
        } else a.resultType match {
          case LongType => row.setLong(j + 1, if (c.longs != null) c.longs(g) else c.ints(g).toLong)
          case IntegerType => row.setInt(j + 1, if (c.ints != null) c.ints(g) else c.longs(g).toInt)
          case DoubleType => row.setDouble(j + 1, c.doubles(g))
        }
        j += 1
      }
      proj(row)
    }
  }
}
