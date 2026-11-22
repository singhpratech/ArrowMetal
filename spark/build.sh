#!/bin/bash
# Build spark/build/arrowmetal-spark.jar, the ArrowMetal plugin for Apache Spark.
#
# Needs: a JDK 22 or newer (the foreign function API; JAVA_HOME), and a Python with pyspark installed
# (its jars/ folder holds Spark's classes and the Scala 2.13 compiler the plugin is compiled with).
#   PYTHON=/path/to/python spark/build.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
JAVA_HOME="${JAVA_HOME:?set JAVA_HOME to a JDK 22 or newer}"
JARS="$("$PYTHON" -c 'import os, pyspark; print(os.path.join(os.path.dirname(pyspark.__file__), "jars"))')"
SCALA_VER="$(ls "$JARS" | sed -n 's/^scala-library-\(.*\)\.jar$/\1/p')"
CP="$(ls "$JARS"/*.jar | tr '\n' ':')"
rm -rf build/classes build/arrowmetal-spark.jar && mkdir -p build/classes
"$JAVA_HOME/bin/javac" --release 22 -d build/classes src/main/java/org/arrowmetal/spark/*.java
"$JAVA_HOME/bin/java" -cp "$JARS/scala-compiler-$SCALA_VER.jar:$JARS/scala-library-$SCALA_VER.jar:$JARS/scala-reflect-$SCALA_VER.jar" \
  scala.tools.nsc.Main -nowarn -classpath "$CP:build/classes" -d build/classes src/main/scala/org/arrowmetal/spark/*.scala
"$JAVA_HOME/bin/jar" cf build/arrowmetal-spark.jar -C build/classes .
echo "built spark/build/arrowmetal-spark.jar (Spark jars from $JARS, Scala $SCALA_VER)"
