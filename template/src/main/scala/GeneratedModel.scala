import core.Expr
import core.compile.CompilerPhase
import core.rewrite.RewriteStep

trait GeneratedModel {

  val name: String

  def generateIR(): Expr

  def extraRewrites(): Seq[(CompilerPhase, RewriteStep)]

  def loadData(folder: String): Map[String, Seq[Seq[Int]]]
}
