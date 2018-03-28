

import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.bytedeco.javacpp.opencv_highgui._

import scala.collection.mutable

object Classification_main {
  val featureVectorsCluster = new mutable.MutableList[String]

  val IMAGE_CATEGORIES_LIST = List("Computers", "crossing", "Cycles", "electronics", "female", "male", "Road", "Rocks", "Sidewalk", "Stairs", "vehicles", "Water")

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName(s"Classification_main")
      .setMaster("local[*]")
      .set("spark.executor.memory", "10g")
      .set("spark.driver.memory", "10g")
    val sparkConf = new SparkConf().setAppName("ImageClassification").setMaster("local[*]").set("spark.driver.memory", "10g").set("spark.executor.memory", "10g")

   // System.setProperty("hadoop.home.dir", "C://winutils")

    val sc=new SparkContext(sparkConf)

    val images_for_train = sc.wholeTextFiles(s"${Pathsettings.INPUT_DIRECTORY}/*/*.JPEG")

    /**
      * Extracts Key Descriptors from the Training set
      * Saves it to a text file
      */
    Descriptors_extraction.extractDescriptors(sc, images_for_train)
    println("*********************Descriptors Extracted*************************")
    /**
      * Reads the Key descriptors and forms a 'K' cluster
      * Saves the centers as a text file
      */
    K_means.kMeansCluster(sc)
    println("*********************K-Means Clustering Done*************************")
    /**
      * Forms a labeled Histogram using the Training set
      * Saves it in the form of label, [Histogram]
      *
      * This shall be used as a input to Random Forest Classifier to create a model
      */
    createHistograms(sc, images_for_train)
    println("*********************Histograms Created*************************")
    /**
      * From the labeled Histograms a Random Forest Model is created
      */
    Random_forest_model.generateRandomForestModel(sc)

    //testImageClassification(sc)

    sc.stop()


//
//    val testImages = sc.wholeTextFiles(s"${Pathsettings.TEST_INPUT_DIRECTORY}/*/*.JPEG")
//    val testImagesArray = testImages.collect()
//    var predictionLabels = List[String]()
//    testImagesArray.foreach(f => {
//      println(f._1)
//      val splitStr = f._1.split("file:/")
//      val predictedClass: Double = Classifyimage.classifyImage(sc, splitStr(1))
//      val segments = f._1.split("/")
//      val cat = segments(segments.length - 2)
//      val GivenClass = IMAGE_CATEGORIES_LIST.indexOf(cat)
//      println(s"Predicting test image : " + cat + " as " + IMAGE_CATEGORIES_LIST(predictedClass.toInt))
//      predictionLabels = predictedClass + ";" + GivenClass :: predictionLabels
//    })
//
//
//
//
//    val pLArray = predictionLabels.toArray
//
//    predictionLabels.foreach(f => {
//      val ff = f.split(";")
//      println(ff(0), ff(1))
//    })
//    val predictionLabelsRDD = sc.parallelize(pLArray)
//
//
//    val pRDD = predictionLabelsRDD.map(f => {
//      val ff = f.split(";")
//      (ff(0).toDouble, ff(1).toDouble)
//    })
//    val accuracy = 1.0 * pRDD.filter(x => x._1 == x._2).count() / testImages.count
//
//    println("Accuracy of the model is" + accuracy)
//    Evaluation_Confusion_matrix.evaluateModel(pRDD)
//
//
//  }




  /**
    *
    * @param sc     : SparkContext
    * @param images : Images list from the training set
    */

  def createHistograms(sc: SparkContext, images: RDD[(String, String)]): Unit = {
    if (Files.exists(Paths.get(Pathsettings.HISTOGRAM_PATH))) {
      println(s"${Pathsettings.HISTOGRAM_PATH} exists, skipping histograms creation..")
      return
    }

    val createdKmeansModel = KMeansModel.load(sc, Pathsettings.KMEANS_PATH)

    val kMeansCenters = sc.broadcast(createdKmeansModel.clusterCenters)

    val imagecategories = sc.broadcast(IMAGE_CATEGORIES_LIST)

    val data = images.map {
      case (name, contents) => {

        val vocabulary = Image_conversions.vectorsToMat(kMeansCenters.value)

        val descriptors = Image_conversions.bowDescriptors(name.split("file:/")(1), vocabulary)
        val list = Image_conversions.matToString(descriptors)
        println("List Size in String " + list.size)


        val segments = name.split("/")
        val cat = segments(segments.length - 2)
        List(imagecategories.value.indexOf(cat) + "," + list(0))
      }
    }.reduce((x, y) => x ::: y)

    val featuresSeq = sc.parallelize(data)

    featuresSeq.saveAsTextFile(Pathsettings.HISTOGRAM_PATH)
    println("Total size of all features : " + data.size)
  }


}