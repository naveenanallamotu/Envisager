import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.tree.model.RandomForestModel

/**
  * Created by nandanamudi on 3/20/17.
  */
object Classifyimage {

 def classifyImage(sc: SparkContext, path: String): Double = {

  val model = KMeansModel.load(sc, Pathsettings.KMEANS_PATH)
  val vocabulary = Image_conversions.vectorsToMat(model.clusterCenters)

  val desc = Image_conversions.bowDescriptors(path, vocabulary)

  val histogram = Image_conversions.matToVector(desc)

  println("--Histogram size : " + histogram.size)

  val nbModel = RandomForestModel.load(sc, Pathsettings.RANDOM_FOREST_PATH)
  //println(nbModel.labels.mkString(" "))

  val p = nbModel.predict(histogram)
  println(s"Predicting test image : " + Classification_main.IMAGE_CATEGORIES_LIST(p.toInt))
  println(s"Predicting test image : " + Classification_main.IMAGE_CATEGORIES_LIST(p.toInt))

  p

 }


}
