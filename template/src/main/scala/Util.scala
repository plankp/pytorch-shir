import scala.io.Source
import java.io.File
import core.Expr
import core.compile.CompilerPhase
import core.rewrite.{RewriteAll, RewriteStep, RewriteTargeted}
import backend.hdl.HDLProject
import backend.hdl.sim.ModelSimExec
import backend.hdl.arch.{ArchCompiler, MapCompiler}
import backend.hdl.arch.device.DeviceSpecificCompiler
import backend.hdl.arch.rewrite.{CleanupRules, FixTimingRules, InputBufferingRules, ParallelizeDotProductRules}
import backend.hdl.arch.mem.{MemFunctionsCompiler, MemoryImage, MemoryLayout}

object Util {

  val usage = """
program [dir] [flags]

Allowed switches and arguments:
  dir               data directory
  -h | --help       display this message
  --gen             generates the necessary hardware files (default)
  --no-gen          opposite of --gen
  --sim             performs simulation with whatever was generated (default)
  --no-sim          opposite of --sim
"""

  def readIntCSV(fname: File): Seq[Seq[Int]] = {
    val f = Source.fromFile(fname, "UTF-8")
    val s = f.getLines().map(_.split(",").map(_.toInt).toList).toList
    f.close()
    s
  }

  def processArgv(args: Array[String]): Option[(Option[String], Boolean, Boolean)] = {
    var inputDir: Option[String] = None
    var genVHDL: Boolean = true
    var simulate: Boolean = true

    for (arg <- args) {
      arg match {
        case "-h" | "--help" =>
          println(usage)
          return None
        case "--gen" =>
          genVHDL = true
        case "--no-gen" =>
          genVHDL = false
        case "--sim" =>
          simulate = true
        case "--no-sim" =>
          simulate = false
        case s =>
          if (s(0) == '-')
            println(s"Warning: ${s} is treated as data directory")
          inputDir = Some(s)
      }
    }
    return Some((inputDir, genVHDL, simulate))
  }

  def drive(model: GeneratedModel, args: Array[String]): Unit = {
    processArgv(args) match {
      case None => ()
      case Some((inputDir, genVHDL, simulate)) =>
        // compute it here instead of going through HDLProject
        // (which would cause IR generation even when unnecessary)
        val projectFolder = s"out/${model.name}"

        if (genVHDL) {
          // also generate the memory layout file so that the Python
          // side can use this information to generate memory images.
          val project = HDLProject(model.name, model.generateIR(), CompilerPhase.first(), model.extraRewrites() ++ Seq(
            (ArchCompiler.phaseBefore, RewriteStep(RewriteAll(), Seq(algo.torch.rewrite.Rules.padInputToCacheline(512)))),
            (ArchCompiler.phaseAfter, RewriteStep(RewriteAll(), Seq(CleanupRules.removeIdentityResizes))),

            (MemFunctionsCompiler.phaseAfter, RewriteStep(RewriteAll(), CleanupRules.get())),
            (DeviceSpecificCompiler.phaseAfter, RewriteStep(RewriteAll(), FixTimingRules.get())),
            (MapCompiler.phaseAfter, RewriteStep(RewriteAll(), CleanupRules.get())),
          ))
          assert(project.PROJECT_FOLDER == projectFolder, "HDLProject is working with a different folder!")
          project.writeHDLFiles()
          MemoryLayout(project.compiledExpr).toFile("memory.layout", projectFolder)
        }

        inputDir match {
          case Some(dir) =>
            val layout = MemoryLayout.fromFile(s"${projectFolder}/memory.layout")
            MemoryImage(layout, model.loadData(dir)).toFiles(projectFolder)
          case None if simulate =>
            println("Warning: performing simulation with potentially outdated or even nonexistent data")
          case None => ()
        }

        if (simulate) {
          // we want to print the result because it has metrics that
          // are useful for us. of course, since we don't always
          // print the "real" Python computed result, the
          // correctresult metric becomes kind of useless...
          val result = ModelSimExec.run(projectFolder)
          result.print()
        }
    }
  }
}
