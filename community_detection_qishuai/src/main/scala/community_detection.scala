import java.io.File

import Louvain.execute
import com.fasterxml.jackson.databind.{ObjectMapper, ObjectWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext, sql}
import org.apache.spark.graphx._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.mutable
import scala.reflect.ClassTag
import org.json4s.native.Json
import org.json4s.DefaultFormats

import scala.collection.mutable.ListBuffer


object community_detection {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    System.setProperty("hadoop.home.dir", "C:\\opt\\spark\\spark-2.2.0-bin-hadoop2.7")
    val conf = new SparkConf().setAppName("community").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder
      .master("local[*]")
      .getOrCreate
    val airportsDF = spark.read.format("csv").option("header", "true").load("./data/airports.csv")
      .withColumnRenamed("Airport ID", "AirportID")
    val routesDF = spark.read.format("csv").option("header", "true").load("./data/routes.csv")
      .withColumnRenamed("Source airport ID", "SrcAirportID")
      .withColumnRenamed("Destination airport ID", "DstAirportID")
    val combinedDF = importData(spark, airportsDF, routesDF)
    val vertices = generateVertics(spark, combinedDF, airportsDF)
    val graph = generateGraph(spark, combinedDF, airportsDF)
    val communities = execute(sc, graph).collect()
    var nodes = new ListBuffer[(Long, Int)]()
    communities.foreach(row => {
      val communitySize = row._2.size
      val node = (row._1, communitySize)
      nodes += node
    })
    val resolvedVertex = sc.parallelize(nodes.toList)
    visualizeCommunities(resolvedVertex, graph, vertices)
  }
  def importData(spark: SparkSession, airportsDF: sql.DataFrame, routesDF: sql.DataFrame): sql.DataFrame = {
    import spark.implicits._
    val combined1 = airportsDF.as("d1").join(routesDF.as("d2"), $"d1.AirportID" === $"d2.SrcAirportID")
      .select("SrcAirportID", "DstAirportID", "Latitude", "Longitude")
      .withColumnRenamed("Latitude", "SrcLatitude")
      .withColumnRenamed("Longitude", "SrcLongitude")
    val combined2 = airportsDF.as("d1").join(combined1.as("d2"), $"d1.AirportID" === $"d2.DstAirportID")
      .select("SrcAirportID", "DstAirportID", "SrcLatitude","SrcLongitude", "Latitude", "Longitude")
      .withColumnRenamed("Latitude", "DstLatitude")
      .withColumnRenamed("Longitude", "DstLongitude")
    val combinedDF = combined2.select(combined2("SrcAirportID").cast("Long").as("SrcAirportID"), combined2("DstAirportID").cast("Long").as("DstAirportID"),
      combined2("SrcLatitude").cast("Double").as("SrcLatitude"), combined2("SrcLongitude").cast("Double").as("SrcLongitude"),
      combined2("DstLatitude").cast("Double").as("DstLatitude"), combined2("DstLongitude").cast("Double").as("DstLongitude"))
    combinedDF
  }
  def generateVertics(spark: SparkSession, combinedDF: sql.DataFrame, airportsDF: sql.DataFrame): RDD[(Long, (String, (Double, Double)))] = {

    val vertexs: RDD[(Long, (String, (Double, Double)))] = airportsDF.select(airportsDF("AirportID").cast("Long").as("AirportID"), airportsDF("IATA"),
      airportsDF("Latitude").cast("Double").as("Latitude"), airportsDF("Longitude").cast("Double").as("Longitude")
    ).rdd
        .map(row => (row.getLong(0), (row.getString(1), (row.getDouble(2),row.getDouble(3)))))

    vertexs

  }

  def generateGraph(spark: SparkSession, combinedDF: sql.DataFrame, airportsDF: sql.DataFrame): Graph[None.type,Double] = {
    val distancesRDD = combinedDF.rdd.map(row => Row(row.getLong(0), row.getLong(1), Distance.geoDistance((row.getDouble(2), row.getDouble(3)), (row.getDouble(4),row.getDouble(5)))))
    val distancesDF = spark.createDataFrame(distancesRDD,
      StructType(
        Seq(
          StructField(name = "SrcAirportID", dataType = LongType, nullable = false),
          StructField(name = "DstAirportID", dataType = LongType, nullable = false),
          StructField(name = "Distance", dataType = DoubleType, nullable = false)
        )
      ))
    val assembler = new VectorAssembler()
      .setInputCols(Array("Distance"))
      .setOutputCol("vectorDistance")

    val assembledDF = assembler.transform(distancesDF)

    val scaler = new MinMaxScaler()
      .setMax(5)
      .setMin(1)
      .setInputCol("vectorDistance")
      .setOutputCol("scaledDistance")
    val scalerModel = scaler.fit(assembledDF)
    // rescale each feature to range [min, max].
    val scaledDistance = scalerModel.transform(assembledDF)

    val edges = scaledDistance.rdd.map(row => Edge(row.getLong(0), row.getLong(1), row.getAs[DenseVector](4).apply(0)))

    Graph.fromEdges(edges,None)

  }

  def visualizeCommunities(resolvedVertex: RDD[(Long, Int)], g:Graph[None.type,Double], vertices: RDD[(Long, (String, (Double, Double)))]): Unit = {

    val geoList: RDD[(Long,(String, (Double, Double)))] = vertices.map(v => (v._1,v._2))

    val geoNodes: Array[JsNode] = resolvedVertex
      .join(geoList)
      .map {
        case (uuid, (value, state)) =>
          (uuid.toInt, state, value)
      }
      .map { geoNode => JsNode(geoNode._2._1, geoNode._1, geoNode._3, s"${geoNode._2._2._1} ${geoNode._2._2._2}") }
      .distinct()
      .collect()


    //Transform edges
    val jsonEdges: Array[JsLink] = g.edges
      .map { edge => JsLink(edge.srcId.toInt, edge.dstId.toInt, edge.attr.toInt) }
      .collect()

    //Output to file
    val mapper = new ObjectMapper()
    val jsGraph = JsGraph(geoNodes, jsonEdges)
    val json = Json(DefaultFormats).write(jsGraph)
     mapper.writeValue(new File("./html/communities.json"), json)
  }
}

case class CommunityNode(nodeID: Long, CommunityID: Long, CommunitySize: Int)
case class JsNode(name: String, Id: Int, value: Int, location: String)
case class JsLink(source: Int, target: Int, value: Int)
case class JsGraph(nodes: Array[JsNode], links: Array[JsLink])