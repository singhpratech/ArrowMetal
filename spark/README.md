# ArrowMetal for Apache Spark

A Spark SQL extension that runs sorts and the partial `GROUP BY` of a Parquet or ORC scan on the Apple
silicon GPU through ArrowMetal. Everything it does not take stays on Spark's own operators.

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
`spark/check.py` compares every answer with Spark's own and times both.
