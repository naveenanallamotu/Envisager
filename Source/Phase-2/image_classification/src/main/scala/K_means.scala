import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors


object K_means {

  def kMeansCluster(sc: SparkContext): Unit = {
    if (Files.exists(Paths.get(Pathsettings.KMEANS_PATH))) {
      println(s"${Pathsettings.KMEANS_PATH} exists, skipping clusters formation..")
      return
    }

    // Load and parse the data
    val features_data = sc.textFile(Pathsettings.FEATURES_PATH)
    val parsedData = features_data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))

    // Cluster the data into {#300} classes using KMeans
    val numClusters = 300
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    //val WSSSE = clusters.computeCost(parsedData)
    //println("Within Set Sum of Squared Errors = " + WSSSE)

    clusters.save(sc, Pathsettings.KMEANS_PATH)
    println(s"Saves Clusters to ${Pathsettings.KMEANS_PATH}")
    sc.parallelize(clusters.clusterCenters.map(v => v.toArray.mkString(" "))).saveAsTextFile(Pathsettings.KMEANS_CENTERS_PATH)
  }

}
