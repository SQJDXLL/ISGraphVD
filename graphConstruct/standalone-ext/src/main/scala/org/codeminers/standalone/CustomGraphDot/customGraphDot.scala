package org.codeminers.standalone.CustomGraphDot

import io.shiftleft.codepropertygraph.generated.EdgeTypes
import io.shiftleft.codepropertygraph.generated.nodes.{AstNode, MethodParameterOut}
import io.shiftleft.semanticcpg.dotgenerator.DotSerializer
import io.shiftleft.semanticcpg.dotgenerator.DotSerializer.{Edge, Graph}
import io.shiftleft.semanticcpg.language._
import overflowdb.traversal._
import org.codeminers.standalone.CustomGraphGenerator.CustomGraphGenerator


class CustomGraphNodeDot[NodeType <: AstNode](val traversal: Traversal[NodeType]) extends AnyVal {

  def dotCustomGraph: Traversal[String] = DotCustomGraphGenerator.DotCustomGraph(traversal)

//  def plotDotAst(implicit viewer: ImageViewer): Unit = {
//    Shared.plotAndDisplay(dotAst.l, viewer)
//  }

}

object DotCustomGraphGenerator {

  def DotCustomGraph[T <: AstNode](traversal: Traversal[T]): Traversal[String] =
    traversal.map(DotCustomGraph)

  def DotCustomGraph(astRoot: AstNode): String = {
    val CustomGraph = new CustomGraphGenerator().generate(astRoot)

    //将原ast生成的部分与新的边进行合并,并且显示时将边的类型也显示出来

    DotSerializer.dotGraph(Option(astRoot), CustomGraph, true)
  }

}

