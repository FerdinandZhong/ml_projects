object Distance {

  def euclideanDistance(a: Array[Double], b: Array[Double]) =
    math.sqrt(a.zip(b).map(p => p._1 - p._2).map(d => d * d).sum)

  def geoDistance(start: (Double, Double), end: (Double,Double)): Double = {
    geoDistance(start._1, start._2, end._1, end._2)
  }

  def geoDistance(startLat: Double, startLong: Double, endLat: Double, endLong: Double): Double = {
    val EARTH_RADIUS = 6371
    val dLat = Math.toRadians(endLat - startLat)
    val dLong = Math.toRadians(endLong - startLong)
    val radStartLat = Math.toRadians(startLat)
    val radEndLat = Math.toRadians(endLat)

    val a = Math.pow(Math.sin(dLat / 2), 2) + Math.cos(radStartLat) * Math.cos(radEndLat) * Math.pow(Math.sin(dLong / 2), 2)
    val c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    EARTH_RADIUS * c
  }
}
