import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.commons.configuration.{ PropertiesConfiguration => HierConf }
import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable._

object InvertedIndex{
  def main(args : Array[String]){
    val conf = new SparkConf().setAppName("invertedIndex")
      .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .set("spark.akka.frameSize","256")
      .set("spark.ui.port","4071")
    val sc = new org.apache.spark.SparkContext(conf)
    val cfg = new HierConf(args(0))
    val inputfile = cfg.getString("inputfile")
    val result = sc.textFile(inputfile)
      .map(x => x.split("\t"))
      .map(x => (x(0), x(1)))
      .map(x => x._2.split(" ").map(y => (y, x._1)))
      .flatMap(x => x)
      .reduceByKey( (x, y) => x + "|" + y)
    result.collect.foreach(println)
    sc.stop()
  }
}
