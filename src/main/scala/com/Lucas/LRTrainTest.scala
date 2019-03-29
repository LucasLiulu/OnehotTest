package com.Lucas

import com.alibaba.fastjson.{JSON, JSONObject}
import org.apache.hadoop.hive.ql.exec.spark.session.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.Logger
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object LRTrainTest {
  val logger: Logger = Logger.getLogger(LRTrainTest.getClass)
  val logPath = "I:\\data\\openSourceDataSets\\lr_test03.json"

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf().setAppName("LRTrainTest")
    var runType: String = "spark"
    if (args.length==0){
      runType = "local"
      sparkConf.setMaster(runType)
    }
    val sc: SparkContext = new SparkContext(sparkConf)
//    spark默认不支持hive，需要重新编译spark后方能使用
//    val sqlContext = new HiveContext(sc)
//    import sqlContext.implicits._
//    创建dataframe需要sql context
    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("WARN")
    val colArray2 = Array("gender", "age", "yearsmarried", "children", "religiousness", "education", "occupation", "rating")
//    val dataDF = sqlContext.read.json(logPath).select($"affairs", $"children", $"religiousness", $"education", $"occupation", $"rating")
//    直接从json文件读取并转换成dataframe
    val dataDF: DataFrame = sqlContext.read.format("json").load(logPath)
//    需要进行onehot编码的特征
    val categoricalColumns = Array("gender", "children")
//    采用pipeline的方式处理机器学习流程
    val statesArray = new ListBuffer[PipelineStage]()
//    开始逐个对需要进行onehot编码的特征编码
    for (cate <- categoricalColumns){
//      先使用stringIndexer建立类别索引（按特征值频次排序，最频繁的label会得到index 0）
      val indexer = new StringIndexer().setInputCol(cate).setOutputCol(s"${cate}Index")
//      再用onehotEncoder将分类变量转换成二进制稀疏向量
      val encoder = new OneHotEncoder().setInputCol(indexer.getOutputCol).setOutputCol(s"${cate}classVec")
//      将转换过程放到pipeline中等待触发
      statesArray.append(indexer, encoder)
    }
    val numericCols = Array("affairs", "age", "yearsmarried", "religiousness", "education", "occupation", "rating")
    val assemblerInputs = categoricalColumns.map(_ + "classVec") ++ numericCols
//    使用 VectorAssembler 将所有特征转换成一个向量
    val assembler = new VectorAssembler().setInputCols(assemblerInputs).setOutputCol("features")
    statesArray.append(assembler)
    val pipeline = new Pipeline()
//    以pipeline的形式运行每个PipelineStage
    pipeline.setStages(statesArray.toArray)
//    根据statesArray中记录的每个操作统计dataframe中特征的统计信息
    val pipelineModel = pipeline.fit(dataDF)
//    针对statesArray中的每个操作，进行真实的转换
    val dataset = pipelineModel.transform(dataDF)
//    dataset.show(false)
//    划分训练集和测试集
    val Array(trainningDF, testDF) = dataset.randomSplit(Array(0.6, 0.4), seed = 123)
    println(s"trainingDF size=${trainningDF.count()}, testDF, size=${testDF.count()}")
//    指定训练集中的特征字段和label字段，开始训练LR模型
    val lrModel = new LogisticRegression().setLabelCol("affairs").setFeaturesCol("features").fit(trainningDF)
//    预测，并重命名label字段
    val predictions = lrModel.transform(testDF).select("affairs","features", "rawPrediction", "probability", "prediction").withColumnRenamed("affairs", "label")
//    predictions.show(false)
//    评价模型离线性能
    val evaluator = new BinaryClassificationEvaluator()
//    指定离线评价指标
    evaluator.setMetricName("areaUnderROC")
    val auc = evaluator.evaluate(predictions)
    println(s"areaUnderROC=${auc}")
    // Get evaluation metrics.
//    val metrics = new BinaryClassificationMetrics(predictions.map(Tuple1.apply(_1 = )))
//    val auROC = metrics.areaUnderROC()
//    val auPR  = metrics.areaUnderPR()

    // 保存训练好的模型
    lrModel.write.overwrite().save(".\\model\\first.model")

    // 在Java/scala程序里，引入spark core，spark mllib等包，加载模型。
    // 特别注意：此处加载时用的是LogisticRegressionModel
    val lrLoadModel = LogisticRegressionModel.load(".\\model\\first.model")
    val predictRaw = lrLoadModel.getClass.getMethod("predictRaw", classOf[Vector]).invoke(lrLoadModel, testDF.select("features").first()(0).asInstanceOf[Vector])  //.asInstanceOf[Vector]
    println("predictRaw: " + predictRaw)
//    使用spark 流式计算作预测
//    val models = sparkSession.sparkContext.broadcast(_model.asInstanceOf[ArrayBuffer[NaiveBayesModel]])
//    val f2 = (vec: Vector) => {
//      models.value.map { model =>
//        val predictRaw = model.getClass.getMethod("predictRaw", classOf[Vector]).invoke(model, vec).asInstanceOf[Vector]
//        val raw2probability = model.getClass.getMethod(raw2probabilityMethod, classOf[Vector]).invoke(model, predictRaw).asInstanceOf[Vector]
//        //model.getClass.getMethod("probability2prediction", classOf[Vector]).invoke(model, raw2probability).asInstanceOf[Vector]
//        raw2probability
//      }
//    }
//    sparkSession.udf.register(name , f2)

//    使用反射的方式调用私有成员对象，加速预测
    val raw2probabilityMethod = if (sc.version.startsWith("2.3")) "raw2probabilityInPlace" else "raw2probability"
    logger.info("raw2probabilityMethod: " + raw2probabilityMethod)
    println("raw2probabilityMethod: " + raw2probabilityMethod)
    val raw2probability = lrLoadModel.getClass.getMethod(raw2probabilityMethod, classOf[Vector]).invoke(lrLoadModel, predictRaw).asInstanceOf[Vector]
    logger.info("raw2probability: " + raw2probability)
    println("raw2probability: " + raw2probability)
    // 得到最终预测类别
    val categoryId = raw2probability.argmax
    logger.info("categoryId: " + categoryId)
    println("categoryId: " + categoryId)
    println()
  }

}
