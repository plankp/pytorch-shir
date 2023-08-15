import core.Expr

trait GeneratedModel {

    val name: String

    def generateIR(): Expr

    def loadData(folder: String): Map[String, Seq[Seq[Int]]]
}
