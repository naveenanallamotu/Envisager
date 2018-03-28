import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


object Descriptors_extraction {

  def extractDescriptors(sc: SparkContext, images: RDD[(String, String)]): Unit = {

    if (Files.exists(Paths.get(Pathsettings.FEATURES_PATH))) {
      println(s"${Pathsettings.FEATURES_PATH} exists, skipping feature extraction..")
      return
    }

    val data = images.map {
      case (name, contents) => {
        val desc = Image_conversions.descriptors(name.split("file:/")(1))
        val list = Image_conversions.matToString(desc)
        println("-- " + list.size)
        list
      }
    }.reduce((x, y) => x ::: y)

    val featuresSequence = sc.parallelize(data)

    featuresSequence.saveAsTextFile(Pathsettings.FEATURES_PATH)
    println("Total size : " + data.size)
  }

}
