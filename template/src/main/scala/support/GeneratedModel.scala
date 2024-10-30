package support

import core.Expr
import core.compile.CompilerPhase
import core.rewrite.RewriteStep

trait GeneratedModel {

  val name: String

  def generateIR(): Expr

  def compilerPhase(): CompilerPhase = CompilerPhase.first()

  def extraRewrites(): Seq[(CompilerPhase, RewriteStep)] = Seq.empty

  def loadData(folder: String): Map[String, Seq[Seq[Int]]]
}
