package org.codeminers.standalone.CustomGraphGenerator

import io.shiftleft.codepropertygraph.generated.EdgeTypes
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.dotgenerator.DotSerializer.{Edge, Graph}
import io.shiftleft.semanticcpg.language._
import overflowdb.Node

import scala.jdk.CollectionConverters._


class CustomGraphGenerator {

//  private val edgeType = EdgeTypes.AST

  def generate(astRoot: AstNode): Graph = {
    def shouldBeDisplayed(v: AstNode): Boolean = !v.isInstanceOf[MethodParameterOut]

    //只是新增边结点不需要更换，所以沿用ast的结点
    val vertices = astRoot.ast.filter(shouldBeDisplayed).l

    //AST边
    val edges = vertices.flatMap(v =>
      v.astChildren.filter(shouldBeDisplayed).map { child =>
        Edge(v, child, edgeType = "AST")
      }
    )
    //CFG边
    val edges_cfg = vertices.flatMap(v =>
      v.astChildren.filter(shouldBeDisplayed).map { child =>
        Edge(v, child, edgeType = "CFG")
      }
    )

    val edges_cdg = vertices.flatMap(v =>
      v.astChildren.filter(shouldBeDisplayed).map { child =>
        Edge(v, child, edgeType = "CDG")
      }
    )


//    def edgesToDisplay(srcNode: StoredNode, visited: List[StoredNode] = List()): List[Edge] = {
//      if (visited.contains(srcNode)) {
//        println("kong")
//        List()
//      } else {
//        //筛选出非叶子结点
//        val children = expand(srcNode).filter(x => vertices.contains(x.dst))
//        println("children",children)
//        //        val children             = expand(srcNode).filter(x => vertices.contains(x.dst))
//        //        val (visible, invisible) = children.partition(x => cfgNodeShouldBeDisplayed(x.dst))
//
//        children.toList.flatMap { n =>
//          println("n.dst",n.dst)
//          edgesToDisplay(n.dst, visited ++ List(srcNode)).map(y =>
//            Edge(srcNode, y.dst, edgeType = EdgeTypes.ComputedFrom))
//        }
//      }
//
//    }
      def edgesToDisplay_computedfrom(srcNode: StoredNode, visited: List[StoredNode] = List()): List[Edge] = {
        if (visited.contains(srcNode)) {
//          println("kong")
          List()
        } else {
          //筛选出非叶子结点
          val children = expand_computedfrom(srcNode).filter(x => vertices.contains(x.dst))
//          println("children",children)
          children.toList
        }

      }

    def edgesToDisplay_lastuse(srcNode: StoredNode, visited: List[StoredNode] = List()): List[Edge] = {
      if (visited.contains(srcNode)) {
//        println("kong")
        List()
      } else {
        //筛选出非叶子结点
        val children = expand_lastuse(srcNode).filter(x => vertices.contains(x.dst))
//        println("children",children)
        children.toList
      }

    }

    def edgesToDisplay_lastwrite(srcNode: StoredNode, visited: List[StoredNode] = List()): List[Edge] = {
      if (visited.contains(srcNode)) {
        //        println("kong")
        List()
      } else {
        //筛选出非叶子结点
        val children = expand_lastwrite(srcNode).filter(x => vertices.contains(x.dst))
        //        println("children",children)
        children.toList
      }

    }

      //    def edge_ComputedFrom():List[Edge] = {
      //      vertices.isIdentifier
      //    }

      val edges_CustomGraph_computedfrom = vertices.isIdentifier.l.flatMap { v =>
//        println("edgesToDisplay(v)",edgesToDisplay_computedfrom(v))
        edgesToDisplay_computedfrom(v)
      }.distinct

    val edges_CustomGraph_lastuse = vertices.isIdentifier.l.flatMap { v =>
//      println("edgesToDisplay(v)",edgesToDisplay_lastuse(v))
      edgesToDisplay_lastuse(v)
    }.distinct

    val edges_CustomGraph_lastwrite = vertices.isIdentifier.l.flatMap { v =>
      //      println("edgesToDisplay(v)",edgesToDisplay_lastuse(v))
      edgesToDisplay_lastwrite(v)
    }.distinct



//    println("edges_CustomGraph",edges_CustomGraph_computedfrom)

      //边合并，节点还是ast中有的节点 有cfg
      // Graph(vertices, edges ++ edges_cfg ++ edges_cdg ++ edges_CustomGraph_computedfrom ++ edges_CustomGraph_lastuse ++ edges_CustomGraph_lastwrite )
//      Graph(vertices, edges ++ edges_cfg ++ edges_CustomGraph_computedfrom ++ edges_CustomGraph_lastuse ++ edges_CustomGraph_lastwrite )
    Graph(vertices, edges ++ edges_CustomGraph_computedfrom ++ edges_CustomGraph_lastuse ++ edges_CustomGraph_lastwrite )

  }

  //通过边类型进行筛选，如果之后新增别的边也可以通过（出边类型：_ComputedFromOut）进行筛选
//  for custom edges
  protected def expand_computedfrom(v: StoredNode): Iterator[Edge] = {
    v._computedfromOut.asScala
      .filter(_.isInstanceOf[AstNode])
      .map(node => Edge(v, node, edgeType = "ComputedFrom"))
  }

  protected def expand_lastuse(v: StoredNode): Iterator[Edge] = {
    v._lastuseOut.asScala
      .filter(_.isInstanceOf[AstNode])
      .map(node => Edge(v, node, edgeType = "LastUse"))
  }

  protected def expand_lastwrite(v: StoredNode): Iterator[Edge] = {
    v._lastwriteOut.asScala
      .filter(_.isInstanceOf[AstNode])
      .map(node => Edge(v, node, edgeType = "LastWrite"))
  }

//  //for cfg
//  protected def expand(v: StoredNode): Iterator[Edge] = {
//    v._cfgOut.asScala
//      .filter(_.isInstanceOf[StoredNode])
//      .map(node => Edge(v, node, edgeType = EdgeTypes.CFG))
//  }



}



