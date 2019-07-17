import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object BasicClassification2 extends App with Logging {
  // True means we are in Development so we need to visualize the neural training, false Production
  val flagVisualizeDevelopmentOrProduction: Boolean = false

  val width: Int = 28 //画像の横幅
  val height: Int = 28 //画像の縦幅
  val nChannels = 1 //入力チャネルの数
  val outputNum = 10 //出力結果の数
  val batchSize = 64 //???
  val nEpochs = 1 //トレーニングする世代数
  val weightDecay: Double = 0.0005 //ウェイトの減衰量
  val iterations = 1 //トレーニングの繰り返し回数
  val seed = 123 //neural networkを初期化する際の乱数のシード
  val scoreFrequency: Int = 100

  // MNIST手書き文字画像DB(http://yann.lecun.com/exdb/mnist/)を学習用データセットとして読み込む準備
  logger.info("Loading MNIST data to training dataset and test dataset...")
  val mnistTrain = new MnistDataSetIterator(batchSize, true, seed)
  val mnistTest = new MnistDataSetIterator(batchSize, false, seed)

  logger.info("Building CNN model...")
  val model = NeuralNet(inputType = InputType.convolutionalFlat(height, width, nChannels), rngSeed = seed)

  model.add(Convolution2D(20, List(5, 5), nChannels, regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(MaxPooling2D(List(2, 2), List(2, 2)))

  model.add(Convolution2D(50, List(5, 5), regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(MaxPooling2D(List(2, 2), List(2, 2)))


  model.add(Dense(512, regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(Dense(outputNum, activation = Activation.SOFTMAX))
  model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)

  logger.info("Train model...")
  model.fit(mnistTrain, nEpochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model...")
  logger.info(s"Train accuracy = ${model.evaluate(mnistTrain).accuracy}")
  logger.info(s"Test accuracy = ${model.evaluate(mnistTest).accuracy}")

  val eval = model.evaluate(mnistTest)
  eval.accuracy()
  eval.precision()
  eval.recall()
  logger.info(eval.stats())
}