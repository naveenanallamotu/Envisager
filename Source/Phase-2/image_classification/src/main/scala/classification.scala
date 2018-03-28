import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.tree.model.RandomForestModel

object classification{
  def classifyImage(path: String): String = {
    val conf = new SparkConf()
      .setAppName(s"Classification_main")
      .setMaster("local[*]")
      .set("spark.executor.memory", "10g")
      .set("spark.driver.memory", "10g")
    val sparkConf = new SparkConf().setAppName("ImageClassification").setMaster("local[*]").set("spark.driver.memory", "10g").set("spark.executor.memory", "10g")

    // System.setProperty("hadoop.home.dir", "C://winutils")

    val sc=new SparkContext(sparkConf)

    val model = KMeansModel.load(sc, Pathsettings.KMEANS_PATH)
    val vocabulary = Image_conversions.vectorsToMat(model.clusterCenters)

    val desc = Image_conversions.bowDescriptors(path, vocabulary)

    val histogram = Image_conversions.matToVector(desc)

    println("--Histogram size : " + histogram.size)

    val nbModel = RandomForestModel.load(sc, Pathsettings.RANDOM_FOREST_PATH)
    //println(nbModel.labels.mkString(" "))

    val p = nbModel.predict(histogram)
    val resstring="The image comes under the class:"+Classification_main.IMAGE_CATEGORIES_LIST(p.toInt)

    println(s"Predicting test image : " + Classification_main.IMAGE_CATEGORIES_LIST(p.toInt))

    sc.stop()
    resstring

  }

}