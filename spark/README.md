# ArrowMetal for Apache Spark

A Spark SQL extension that runs Spark's work on the Apple silicon GPU through ArrowMetal: the filters,
projections and partial aggregate directly above a Parquet scan as one GPU stage. Sorts, aggregates
without `GROUP BY` and plain filter/projection stages are opt-in.
Everything it does not take stays on Spark's own operators.

```bash
JAVA_HOME=<a JDK 22 or newer> PYTHON=<a python with pyspark> spark/build.sh     # spark/build/arrowmetal-spark.jar
```

```bash
spark-submit \
  --jars spark/build/arrowmetal-spark.jar \
  --conf spark.sql.extensions=org.arrowmetal.spark.ArrowMetalExtensions \
  --conf spark.driver.extraJavaOptions=--enable-native-access=ALL-UNNAMED \
  --conf spark.executorEnv.ARROWMETAL_LIB=/path/to/libArrowMetalC.dylib \
  ...
```

What it covers, how it reaches the GPU and what was measured: [docs/SPARK.md](../docs/SPARK.md).
`spark/fuzz.py` runs random queries with and without the plugin and compares the answers;
`spark/check.py` checks and times the sort, the `GROUP BY` and whole stages.
