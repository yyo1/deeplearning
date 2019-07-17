name := "deeplearning"

version := "0.1"

scalaVersion := "2.11.7"

val nd4jVersion = "1.0.0-alpha"
libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % nd4jVersion,
  "org.deeplearning4j" %% "deeplearning4j-ui" % nd4jVersion,
  "org.deeplearning4j" % "scalnet_2.11" % nd4jVersion,
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion classifier "macosx-x86_64",
  "org.nd4j" %% "nd4s" % nd4jVersion,
  "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.2"
  //"org.bytedeco" % "javacv-platform" % "1.3.2"
)
