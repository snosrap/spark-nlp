/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec
import scala.language.reflectiveCalls


trait DependencyParserBehaviors { this: AnyFlatSpec =>


  def initialAnnotations(testDataSet: Dataset[Row]): Unit = {
    val fixture = createFixture(testDataSet)
    it should "add annotations" taggedAs FastTest in {
      assert(fixture.dependencies.count > 0, "Annotations count should be greater than 0")
    }

    it should "add annotations with the correct annotationType" taggedAs FastTest in {
      fixture.depAnnotations.foreach { a =>
        assert(a.annotatorType == AnnotatorType.DEPENDENCY, s"Annotation type should ${AnnotatorType.DEPENDENCY}")
      }
    }

    it should "annotate each token" taggedAs FastTest in {
      assert(fixture.tokenAnnotations.size == fixture.depAnnotations.size, s"Every token should be annotated")
    }

    it should "annotate each word with a head" taggedAs FastTest in {
      fixture.depAnnotations.foreach { a =>
        assert(a.result.nonEmpty, s"Result should have a head")
      }
    }

    it should "annotate each word with the correct indexes" taggedAs FastTest in {
      fixture.depAnnotations
        .zip(fixture.tokenAnnotations)
        .foreach { case (dep, token) => assert(dep.begin == token.begin && dep.end == token.end, s"Token and word should have equal indixes") }
    }
  }

  private def createFixture(testDataSet: Dataset[Row]) = new {
    val dependencies: DataFrame = testDataSet.select("dependency")
    val depAnnotations: Seq[Annotation] = dependencies
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
    val tokens: DataFrame = testDataSet.select("token")
    val tokenAnnotations: Seq[Annotation] = tokens
      .collect
      .flatMap { r => r.getSeq[Row](0) }
      .map { r =>
        Annotation(r.getString(0), r.getInt(1), r.getInt(2), r.getString(3), r.getMap[String, String](4))
      }
  }

  def relationshipsBetweenWordsPredictor(testDataSet: Dataset[Row], pipeline: Pipeline): Unit = {

    val emptyDataSet = PipelineModels.dummyDataset

    val dependencyParserModel = pipeline.fit(emptyDataSet)

    it should "train a model" taggedAs FastTest in {
      val model = dependencyParserModel.stages.last.asInstanceOf[DependencyParserModel]
      assert(model.isInstanceOf[DependencyParserModel])
    }

    val dependencyParserDataFrame = dependencyParserModel.transform(testDataSet)
    //dependencyParserDataFrame.collect()
    //dependencyParserDataFrame.select("dependency").show(false)

    it should "predict relationships between words" taggedAs FastTest in {
      assert(dependencyParserDataFrame.isInstanceOf[DataFrame])
    }

  }

}
