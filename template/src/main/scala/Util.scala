import scala.io.Source
import java.io.File
import core.Expr
import backend.hdl.HDLProject
import backend.hdl.sim.ModelSimExec
import backend.hdl.arch.mem.{MemoryImage, MemoryLayout}

object Util {

    val usage = """
program [dir] [flags]

Allowed switches and arguments:
  dir               data directory
  -h | --help       display this message
  --gen             generates the output directory and VHDL files (default)
  --no-gen          opposite of --gen
  --sim             performs simulation with existing VHDL files (default)
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
                val project = HDLProject(model.name, model.generateIR())
                lazy val layout = MemoryLayout(project.compiledExpr)

                if (genVHDL)
                    project.writeHDLFiles()
                inputDir match {
                    case Some(dir) =>
                        MemoryImage(layout, model.loadData(dir)).toFiles(project.PROJECT_FOLDER)
                    case None if simulate =>
                        println("Warning: performing simulation with bogus or nonexistent data")
                    case None =>
                        ()
                }
                if (simulate) {
                    // we want to print the result because it has metrics that
                    // are useful for us. of course, since we don't always
                    // print the "real" Python computed result, the
                    // correctresult metric becomes kind of useless...
                    val result = ModelSimExec.run(project.PROJECT_FOLDER)
                    result.print()
                }
        }
    }
}
