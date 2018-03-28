import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.tree.model.RandomForestModel


object Predict {

  val IMAGE_CATEGORIES_LIST = List("Computers", "crossing", "Cycles", "electronics", "female", "male", "Road", "Rocks", "Sidewalk", "Stairs", "vehicles", "Water")

  def testImageClassification(sc: SparkContext, path: String): String ={

    val model = KMeansModel.load(sc, Pathsettings.KMEANS_PATH)
    val vocabulary = Image_conversions.vectorsToMat(model.clusterCenters)
    val desc = Image_conversions.bowDescriptors(path, vocabulary)
    val histogram = Image_conversions.matToVector(desc)

    println("-- Histogram size : " + histogram.size)
    println(histogram.toArray.mkString(" "))

    val nbModel = RandomForestModel.load(sc, Pathsettings.RANDOM_FOREST_PATH)
    val p = nbModel.predict(histogram)
    (s"Test image predicted as : " + IMAGE_CATEGORIES_LIST(p.toInt))

  }


  def testImage(string: String):String = {
    System.setProperty("hadoop.home.dir", "C://winutils")
    val conf = new SparkConf()
      .setAppName(s"ImageClassificatin")
      .setMaster("local[*]")
      .set("spark.executor.memory", "6g")
      .set("spark.driver.memory", "6g")

    val sparkConf = new SparkConf().setAppName("ImageClassificatin").setMaster("local[*]")
    val sc= SparkContext.getOrCreate(sparkConf)
    val res = testImageClassification(sc, string)

    printf(res);
    res
  }
}
