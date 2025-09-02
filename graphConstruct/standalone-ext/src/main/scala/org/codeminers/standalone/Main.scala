package org.codeminers.standalone
import java.io._
import io.joern.x2cpg.X2Cpg.applyDefaultOverlays
 
import io.shiftleft.codepropertygraph.generated.{Cpg, EdgeTypes}
import io.shiftleft.codepropertygraph.generated.nodes.AstNode
import overflowdb.{Edge, Graph, Node, Property, PropertyKey}
import scala.util.control.Breaks
import java.util
import java.util.Optional
import java.nio.file.{Files, Path, Paths}

import scala.collection.mutable
 
import io.shiftleft.passes.SimpleCpgPass
import io.shiftleft.semanticcpg.language._
import org.checkerframework.checker.signature.qual.Identifier
import overflowdb.BatchedUpdate
 
import io.circe._
import io.circe.generic.auto._
import io.circe.syntax._

import scala.xml.{NodeSeq, XML}
import scala.xml.factory.XMLLoader
import scala.util.Try

import scala.util.{Failure, Success}
import scala.util.control.Breaks._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Stack
import scala.collection.mutable.Map

import scala.xml.XML

import io.joern.c2cpg.{Config,C2Cpg}
import io.joern.x2cpg

import org.codeminers.standalone.CustomGraphDot.CustomGraphNodeDot


object Main extends App {
  println("Received arguments: " + args.mkString(", "))

  args.foreach(println)
  val variableType = args.getClass.getName

  val startTime = System.currentTimeMillis()


  val currentPath = Paths.get(".").toAbsolutePath.normalize.toString


  println("args.lengrh", args.length)
  val file_path: String = if (args.length > 3) { 
    "../data_detect/data/" + args(3) + "/" + args(0) + "/" + args(1) + "/pseudo/"  
  } else {  
    "../data/" + args(0) + "/" + args(1) + "/pseudo/"  
  }

  val directory_src:File = new File(file_path)
  println("directory_src", directory_src)

  def getFiles1(dir: File): Array[File] = {
    val files = dir.listFiles
    print("files", files)
    if (files != null) {
      files.filter(_.isFile) ++ files.filter(_.isDirectory).flatMap(getFiles1)
    } else {
      Array.empty[File]
    }
  }

  var file_Array = getFiles1(directory_src)
  println("Number of files:", file_Array.length)
  
  for(file_code <- file_Array){
    breakable {
      if (file_code.getName == ".DS_Store") {
        break()
      }

      val directory = directory_src.getAbsolutePath + '/' + file_code.getName()
      println("directory_src.getName()", directory)
      print("directory_src.getAbsolutePath ", directory_src.getAbsolutePath )
      var name_of_input = file_code.getName()
      val elements = name_of_input.split("\\.")  
      val fileNameWithoutExtension = elements.init.mkString(".")
      val cpgbin_newpath = Paths.get(directory_src.getAbsolutePath).getParent + "/cpg.bin"
      val config = Config(inputPaths = Set(directory), outputPath = cpgbin_newpath)
      val cpgOrException = new C2Cpg().createCpg(config)
      

      cpgOrException match {
        case Success(cpg) =>
          println("[DONE]")
          println("Applying default overlays")

          applyDefaultOverlays(cpg)
          cpg.method.name.foreach(println)
          
          println("=====================")

          var astroot: AstNode = cpg.method.l(1)

          println("Running a custom pass to add some custom nodes")
          new MyPass(cpg, astroot).createAndApply()

          val CustomGraph2Dot = new CustomGraphNodeDot(cpg.method(cpg.method.name.l(1))).dotCustomGraph.l

          val dotContent_AST: String = CustomGraph2Dot.mkString("\n")
          
          val dotContent_CFG : String = cpg.method(cpg.method.name.l(1)).dotCfg.l.mkString("\n")

          val outputDir: String = if (args.length>3) {  
            "../data_detect/data/" + args(3) + "/" + args(0) + "/" + args(1) + "/graph/" + fileNameWithoutExtension + "/"  
          } else {  
            "../data/" + args(0) + "/" + args(1) + "/graph/" + fileNameWithoutExtension + "/"  
          }

          val outputFileAst = outputDir + "ast_deform.dot"
          val outputFileCfg = outputDir + "CFG.dot"

          import java.nio.file.{Files, Paths}

          val directory_output = new File(outputDir)
          if (!directory_output.exists()) {
            if (directory_output.mkdirs()) {
              println(s"Path '$outputDir' created successfully.")
            } else {
              println(s"Failed to create path '$outputDir'.")
            }
          } else {
            println(s"Path '$outputDir' already exists.")
          }

          def writeToFile(filePath: String, content: String): Unit = {
            val writer = new BufferedWriter(new FileWriter(filePath))
            try {
              writer.write(content)
            } finally {
              writer.close()
            }
          }

          writeToFile(outputFileAst, dotContent_AST)
          writeToFile(outputFileCfg, dotContent_CFG)

        case Failure(exception) =>
          println("[FAILED]")
          println(exception)
      }

    }
  }

  val endTime = System.currentTimeMillis()
  println("endTime", endTime)
  val executionTime = endTime - startTime
}


/** Example of a custom pass that creates and stores a node in the CPG.
  */
class MyPass(cpg: Cpg, astroot:AstNode) extends SimpleCpgPass(cpg) {

  def add_computedFrom(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    var list_assignment:List[String] = List("<operator>.assignment","<operator>.assignmentPlus","<operator>.assignmentMinus",
                                            "<operator>.assignmentDivision","<operator>.assignmentMultiplication","<operator>.assignmentModulo",
                                            "<operator>.assignmentShiftLeft","<operator>.assignmentArithmeticShiftRight","<operator>.assignmentAnd",
                                            "<operator>.assignmentXor","<operator>.assignmentOr")
    cpg.call.foreach(n => {
      if (list_assignment.contains(n.name)){
        var ast_depth = n.depth
        var computedfrom_out_node = n.astChildren.l(0)
        var list_innodes = new ListBuffer[AstNode]()
        for(b <- computedfrom_out_node.ast.l){
          if(b.isIdentifier){
            list_innodes += b
          }
        }

        var list_outnodes = new ListBuffer[AstNode]()
        var n_tmp = n.astChildren.l(1)
        for(a <-n_tmp.ast.l){
          if(a.isIdentifier){
            list_outnodes += a
          }
        }

        for(in <- list_innodes){
          for(out <- list_outnodes){
            builder.addEdge(out, in, "ComputedFrom")
          }
        }
        for ( i <- 1 to list_outnodes.length-1){
          builder.addEdge(list_outnodes(0), list_outnodes(i), "ComputedFrom")
        }
      }
    })
  }

  def add_lastwrite(builder: BatchedUpdate.DiffGraphBuilder): Unit = {

    cpg.call.foreach(n => {
      if (n.name == "<operator>.assignment") {
        var lastwrite_in = n.astChildren.l(0)
        var lastwrite_in_data = lastwrite_in.code
        var n_tmp = n.ast.l(0)
        var flag_continue = true
        var flag_for = true
        breakable({
          if(n.astParent.astParent.isControlStructure == true || n.astParent.isControlStructure == true){
            if(n.astParent.astParent.isControlStructure == true){
              var num_assigment = 0
              var loop_rootnode = n.astParent.astParent
              var list_assignment = new ListBuffer[AstNode]()
              for(k <- n.astParent.astParent.ast.l){
                if (k.code.contains("=") && k.astChildren.l(0).code.contains(lastwrite_in_data) && k.astChildren.l(1).code.contains(lastwrite_in_data)){
                  num_assigment += 1
                  list_assignment += k
                }
              }
              if(n == list_assignment.last){
                var flag_loop = true
                for(l <- loop_rootnode.ast.l if flag_loop){
                  if(l.code == lastwrite_in_data){
                    builder.addEdge(l,lastwrite_in,"LastWrite")
                  }
                  if(l.code.contains("=")){
                    builder.addEdge(lastwrite_in,l.astChildren.l(0),"LastWrite")
                    flag_loop = false
                  }
                }
              }
            }else{
            }
          }
          while(flag_continue == true) {
            for (i <- n_tmp.astParent.ast.l.tail if flag_for) {
              if(i.id > n_tmp.id){
                if(i.code == lastwrite_in_data && i != lastwrite_in){
                  builder.addEdge(lastwrite_in,i,"LastWrite")
                }else if(i.code.contains("=")){
                  for(j <- i.astChildren.l(1).ast.l){
                    if(j.code == lastwrite_in_data){
                      builder.addEdge(lastwrite_in,j,"LastWrite")
                    }
                  }
                  break()
                }
              }
            }
            n_tmp = n_tmp.astParent
          }
        })
      }
    })
  }

  def add_lastuse(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    var list_identifier_use = new ListBuffer[AstNode]()
    cpg.identifier.foreach(n=>{
      list_identifier_use += n
    })

    breakable({
      var map_newuse = new mutable.HashMap[String,mutable.Stack[AstNode]]()
      var map_newuse_stack = new mutable.HashMap[String,mutable.Stack[(AstNode,ListBuffer[AstNode],ListBuffer[AstNode])]]()
      var map_last_stack = new mutable.HashMap[String,mutable.Stack[ListBuffer[(AstNode,AstNode)]]]()
      var map_For_Inner = new mutable.HashMap[AstNode,ListBuffer[String]]()
      var map_For_First = new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      var map_For_Second = new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      var map_For_Outer =  new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      var map_For_Record = new mutable.HashMap[AstNode,ListBuffer[Boolean]]()
      def next_same_name(p: Int) :Int = {
        var num_true_next = p
        if(num_true_next == list_identifier_use.length-1){
          num_true_next = -1
        }else{
          num_true_next = p+1
          breakable({
            while(list_identifier_use(p).code != list_identifier_use(num_true_next).code){
              num_true_next += 1
              if(num_true_next == list_identifier_use.length){
                num_true_next = -1
                break
              }
            }
          })
        }
        num_true_next
      }

      def last_same_name(p:Int) :Int = {
        var num_true_last = p-1
        if(num_true_last <= 0){
          num_true_last = -1
        }else{
          if(list_identifier_use(num_true_last) == astroot){
            num_true_last = -1
          }
          else{
            breakable({
              while(list_identifier_use(p).code != list_identifier_use(num_true_last).code){
                if(list_identifier_use(num_true_last) == astroot){
                  num_true_last = -1
                  break
                }
                num_true_last -=1
                if(num_true_last<1){
                  num_true_last = -1
                  break
                }else{
                  if(list_identifier_use(num_true_last) == astroot){
                    num_true_last = -1
                    break
                  }
                }
              }
            })
          }
        }
        num_true_last
      }

      def same_maxid_parent(a :AstNode,b :AstNode):AstNode ={
        var list_identifier_now_parent = new ListBuffer[AstNode]()
        var list_identifier_next_parent = new ListBuffer[AstNode]()
        var now_parent = a.astParent
        var next_parent = b.astParent
        while(now_parent.id > cpg.method.l(1).id){
          list_identifier_now_parent += now_parent
          now_parent = now_parent.astParent
        }
        while(next_parent.id > cpg.method.l(1).id){
          list_identifier_next_parent += next_parent
          next_parent = next_parent.astParent
        }
        var flag_1 = true
        var same_parent_maxid = a
        for(parent_now <- list_identifier_now_parent if flag_1){
          for(parent_next <- list_identifier_next_parent){
            if(parent_now == parent_next){
              same_parent_maxid = parent_next
              flag_1 = false
            }
          }
        }
        same_parent_maxid
      }

      def is_judge(p: Int) : (String,AstNode) = {
        var judge_for_belong:AstNode = list_identifier_use(p)
        var belong_which:String = "null"
        var belong_which_node:AstNode = judge_for_belong
        breakable{
          while(! judge_for_belong.astParent.isControlStructure){
            judge_for_belong = judge_for_belong.astParent
            if(judge_for_belong.l.isEmpty){
              belong_which = "false"
              break
            }else if(judge_for_belong.code.contains("<empty>")){
              belong_which = "false"
 
              if(judge_for_belong.astParent.code.contains("for") && judge_for_belong.astParent.isControlStructure && judge_for_belong.astParent.code.startsWith("for")){
                if(for_Belong_Which(list_identifier_use(p),judge_for_belong.astParent) != 0){
                  belong_which = "for"
                }
              }
              break
            }
          }
        }
        if(belong_which == "false"){
          belong_which = "false"
        }else{
          belong_which_node = judge_for_belong.astParent
          if(belong_which_node.code.contains("if") || belong_which_node.code.contains("else")){
            belong_which = "if"
          }
          else if(belong_which_node.code.contains("while")){
            if(belong_which_node.code.contains("do")){
              belong_which = "false"
            }
            else{
              belong_which = "while"
            }

          }
          else if(belong_which_node.code.contains("for") && belong_which_node.code.startsWith("for")){
            belong_which = "for"
          }
          else if(belong_which_node.code.contains("switch")){
            belong_which = "switch"
          }
        }
        (belong_which,belong_which_node)
      }

      def is_out_control(p:Int) :ListBuffer[(AstNode,String)] = {
        var find_parent:AstNode = list_identifier_use(p)
        var control_type:String = "true"
        var control_type_node = find_parent
        var num_loop:Int = 0
        var last_control:AstNode = control_type_node
        var control_type_map = new ListBuffer[(AstNode,String)]
        breakable{
          while(control_type != "false" && find_parent != astroot && find_parent.id > astroot.id){
            if(find_parent.l(0).id > astroot.id){
              breakable({
                while(! find_parent.astParent.isControlStructure){
                  find_parent = find_parent.astParent
                  if(find_parent.l(0) == astroot || find_parent.l(0).id < astroot.id){ 
                    control_type = "false"
                    break
                  }
                }
              })
            }
            if(control_type == "false" && num_loop == 0){
              control_type_node = find_parent
              control_type_map += ((control_type_node,control_type))
            }
            else{
              var next_node:AstNode = list_identifier_use(p)
              if(next_same_name(p) != -1){
                next_node = list_identifier_use(next_same_name(p))
              }
              find_parent = find_parent.astParent
              control_type_node = find_parent
              if(find_parent.code.contains("if") || find_parent.code.contains("else")){
                if(find_parent.code.contains("if")){ 
                  if(num_loop > 0 && order_or_nest_plus(find_parent,last_control) == 5)
                  {
                    break() 
                  }
                  else
                  {
                    if(next_same_name(p) == -1){
                      control_type = "if"
                      last_control = control_type_node
                      control_type_map += ((control_type_node,control_type))
                      num_loop +=1
                    }
                    else if(find_parent.astChildren.l(0).ast.l.contains(next_node) || find_parent.astChildren.l(1).ast.l.contains(next_node)){
                      control_type = "false"
                    }
                    else{
                      control_type = "if"
                      last_control = control_type_node
                      control_type_map += ((control_type_node,control_type))
                      if(find_parent.astParent.astParent.code.contains("else")){
                        if(find_parent.astParent.astChildren.l.length == 1) {
                          find_parent = find_parent.astParent.astParent.astParent
                        }
                      }
 
                      num_loop +=1
                    }
                  }
                }
                else
                {
                  if(num_loop > 0 && !is_father(find_parent,last_control)){
                    break() 
                  }
                  else{
                    if(next_same_name(p) == -1){
                      control_type = "if"
                      last_control = control_type_node
                      control_type_map += ((control_type_node,control_type))
                      num_loop += 1
                    }
                    else if(find_parent.astChildren.l(0).ast.l.contains(next_node)){
                      control_type = "false"
                    }
                    else{
                      control_type = "if"
                      last_control = control_type_node
                      control_type_map += ((control_type_node,control_type))

                      num_loop +=1
                    }
                  }
                }
              }
              else if(find_parent.code.contains("for") && find_parent.code.startsWith("for")){
                if(next_same_name(p) == -1){
                  control_type = "for"
                  last_control = control_type_node
                  control_type_map += ((control_type_node,control_type))
                  num_loop +=1
                }
                else if(find_parent.ast.l.contains(next_node)){
                  control_type = "false"
                }else{
                  control_type = "for"
                  last_control = control_type_node
                  control_type_map += ((control_type_node,control_type))
                  num_loop +=1
                }
              }
              else if(find_parent.code.contains("while")){
                if(find_parent.code.contains("do")){ 
                  if(next_same_name(p) == -1){
                    control_type = "do-while"
                    last_control = control_type_node
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                  else if(find_parent.astChildren.l(0).ast.l.contains(next_node) || find_parent.astChildren.l(1).ast.l.contains(next_node)){  
                    control_type = "false"
                  }else{
                    control_type = "do-while"
                    last_control = control_type_node
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                }else{  
                  if(next_same_name(p) == -1){
                    control_type = "while"
                    last_control = control_type_node 
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                  else if(find_parent.ast.l.contains(next_node)){
                    control_type = "false"
                  }else{
                    control_type = "while"
                    last_control = control_type_node 
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                }
              }
              else if(find_parent.code.contains("switch")){
                 
                if(next_same_name(p) == -1){
                  control_type = "switch"
                   
                  last_control = find_belong_case(p,control_type_node)
                  control_type_map += ((last_control,control_type))
                  num_loop +=1
                }
                else{
                   
                   
                  if(! find_parent.astChildren.l(0).ast.l.contains(list_identifier_use(p))){
                    if(find_parent.astChildren.l(1).ast.l.contains(next_node)){
                       

                      val list_range = new ListBuffer[AstNode]()
                       
                      breakable{
                        for (i<-find_parent.astChildren.l(1).ast.l){  
                          if(i.id > list_identifier_use(p).id && i.id < next_node.id){ 
                             
                             
                            if(i.code.contains("case") || i.code.contains("default")){
                              if(i.astParent.astParent == find_parent){
                                 
                                 
                                control_type = "switch"
                                 
                                 
                                 
                                last_control = find_belong_case(p,i.astParent.astParent) 
 
                                control_type_map += ((last_control,control_type))
                                num_loop +=1
                                break
                              }
                            }
                          }
                        }
                      }
                    }
                    else{
                      control_type = "switch"
                      last_control = control_type_node 
                      control_type_map += ((control_type_node,control_type))
                      num_loop += 1
                    }
                  }
                }
                if(control_type != "switch"){ 
                  control_type = "false"
                }
              }
              if(control_type == "false" && num_loop == 0){
                control_type_node = find_parent
                control_type_map += ((control_type_node,control_type))
              }
            }
          }
        }
        control_type_map
      }

      def find_belong_case(p:Int,case_node:AstNode) : AstNode = {
        var node_p : AstNode = list_identifier_use(p)
        var find_node :AstNode = node_p

         
        if(find_some_node(case_node,node_p,"case")._1){
          find_node = find_some_node(case_node,node_p,"case")._2.last
        }
        if(find_some_node(case_node,node_p,"default")._1){
          find_node = find_some_node(case_node,node_p,"case")._2.last
        }

        find_node
      }

      def assignment_right_have(p:Int):Boolean = {
        var is_assignment_right_have: Boolean = false
        if(next_same_name(p) != -1 && next_same_name(next_same_name(p)) != -1){
          if(same_maxid_parent(list_identifier_use(next_same_name(p)),list_identifier_use(next_same_name(next_same_name(p)))).code.contains("=")){
            is_assignment_right_have = true
          }
        }
        is_assignment_right_have
      }

      def left_right(p:Int):Boolean = {
        var id_left: Long = list_identifier_use(p).astParent.astChildren.l(0).id
        var is_left :Boolean = true
        if(list_identifier_use(p).id != id_left){
          is_left = false
        }
        is_left
      }

      def update_data(p:Int): (Boolean,Int) = {
        var update_data:Boolean = true
        var find_assignment:Boolean = false
         
         
        var p_node: AstNode = list_identifier_use(next_same_name(p)).astParent
        var false_type:Int = 0  
        var p_id:Long = list_identifier_use(next_same_name(p)).id
        breakable{ 
          while(p_node != astroot && p_node.id > astroot.id){
             
            if(((p_node.code.contains("=") && !p_node.code.contains("=="))|| p_node.code.contains("++") || p_node.code.contains("--")) && ! p_node.isControlStructure){
              find_assignment = true
              break
            }
             
            p_node = p_node.astParent
          }
        }

        if(find_assignment){
           
          if(p_id >= p_node.astChildren.l.last.id){
             
            update_data = false  
            false_type = 1
            var List_assignment_Self = ListBuffer[String]("++","--")
            breakable{
              for(type_string <- 0 to List_assignment_Self.length-1){
                 
                if(p_node.code.contains(List_assignment_Self(type_string))){ 
                  update_data = false  
                  false_type = 3
                  break
                }
              }
            }
          }else{
            if(next_same_name(p) != -1 && next_same_name(next_same_name(p)) != -1){
               
              if(same_maxid_parent(list_identifier_use(next_same_name(p)),list_identifier_use(next_same_name((next_same_name(p))))) == p_node){
                update_data = false  
                false_type = 2
              }
              else{
                update_data = true  
                var List_assignment = ListBuffer[String]("+=","-=","*=","/=")
                breakable{
                  for(type_string <- 0 to List_assignment.length-1){
                    if(p_node.code.contains(List_assignment(type_string)) && p_id < p_node.astChildren.l.last.id){ 
                      update_data = false  
                      false_type = 3
                      break
                    }
                  }
                }
              }
            }
            else{
              update_data = true  
              var List_assignment = ListBuffer[String]("+=","-=","*=","/=")
              breakable{
                for(type_string <- 0 to List_assignment.length-1){
                  if(p_node.code.contains(List_assignment(type_string)) && p_id < p_node.astChildren.l.last.id){ 
                    update_data = false  
                    false_type = 3
                    break
                  }
                }
              }
            }
          }
        }else{
           
          update_data = false  
          false_type = 4
        }
         
        (update_data,false_type)
      }

      def is_father(p:AstNode, q:AstNode):Boolean = {
        var q_tmp:AstNode = q
        var is_father:Boolean = false
 
        if(p.id < q.id){  
          breakable{
            while(p != q_tmp){
 
              q_tmp = q_tmp.astParent
               
              if(q_tmp.id < p.id){
                break
              }
            }
            is_father = true
          }
        }
        is_father
      }

      def update_map_newuse(i:Int,node:AstNode) :Unit ={
        if (map_newuse.get(list_identifier_use(i).code) == None) {
          map_newuse(list_identifier_use(i).code) = Stack(node)
        } else {
          map_newuse(list_identifier_use(i).code).pop()
          map_newuse(list_identifier_use(i).code).push(node)
        }
      }

      def draw_line(i:Int,node:AstNode) :Unit ={
        if (map_newuse.get(list_identifier_use(i).code) == None || map_newuse(list_identifier_use(i).code).isEmpty) {
          builder.addEdge(node, list_identifier_use(i), "LastUse")
        } else {
          builder.addEdge(node, map_newuse(list_identifier_use(i).code).head ,"LastUse")
        }
      }


      def find_if_father(p:AstNode):Boolean = {
        var is_nested : Boolean = false
        var p_father:AstNode = p.astParent
        breakable{
          while(p_father != astroot){
            if(p_father.code.contains("if") && (p_father.astParent.astChildren.l(2) == p_father)){
              is_nested = true
              break
            }
            p_father = p_father.astParent
          }
        }
        is_nested
      }


      def find_nearest_control(p:Int): AstNode = {
        var node:AstNode = list_identifier_use(p)
        var result:AstNode = node.astParent
        breakable{
          while(result != astroot){
            if(result.isControlStructure){
              break
            }
            result = result.astParent
          }
        }
        result
      }


      def order_or_nest(p:AstNode,q:AstNode) : Boolean = {
        var order_or_nest : Boolean = false
        var p_children_option:Option[AstNode] = p.astChildren.l.lastOption
        if(!p_children_option.isEmpty){
          var p_children:AstNode = p.astChildren.l.last
           
          breakable{
            while(p_children.astChildren.l.lastOption != None){
              if(q.id < p_children.id){  
                break
              }
              else if(p_children == q){
                order_or_nest = true
                break
              }
              p_children = p_children.astChildren.l.last
            }
          }
        }
        order_or_nest
      }

      def order_or_nest_if(p:AstNode,q:AstNode) : Int = {

        var relation :Int = 0
        if(is_father(p,q)){
          var p_children_option:Option[AstNode] = p.astChildren.l.lastOption
          var num_loop : Int = 0
          if(!p_children_option.isEmpty){
            var p_children:AstNode = p.astChildren.l.last
             
            breakable{
              while(p_children.astChildren.l.lastOption != None){
                if(q.id < p_children.id){  
                  if(num_loop == 0){
                     
                    relation = 1
                    break
                  }else{
                     
                    relation = 4
                    break
                  }
                }
                else if(p_children == q){
                  relation = 5
                  break
                }else{
                   
                  relation = 1
                }
                num_loop += 1
                p_children = p_children.astChildren.l.last
              }
            }
          }
        }else{
           
          relation = 5
        }
        relation
      }

      def order_or_nest_switch(p:AstNode,q:AstNode) : Int = {
        var relation : Int = 0
        var p_q_parent : AstNode = same_maxid_parent(p,q)
        var p_switch : AstNode = p.astParent.astParent
        var q_switch : AstNode = q.astParent.astParent
        if(p_switch == q_switch){
           
          relation = 5  
        }
        else{
          var is_find : Boolean = false
          if(is_father(p_switch,q_switch)){
             
             
            for(node <- find_some_node(p,q,"case")._2){
              if(node.astParent.astParent == p_switch){
                is_find = true
              }
            }
            for(node <- find_some_node(p,q,"default")._2){
              if(node.astParent.astParent == p_switch){
                is_find = true
              }
            }
            if(is_find){
              relation = 4  
            }
            else{
              relation = 1  
            }
          }else{
             
            relation = 2  
          }
        }
        relation
      }

      def order_or_nest_plus(p:AstNode,q:AstNode) : Int = {
        var relation : Int = 0
        var p_branch : Boolean = true  
        var q_branch : Boolean = true  


         
        def multi_Or_Single(node:AstNode):Boolean = {
          var multi_or_single: Boolean = true
          if((node.code.contains("for") && node.code.startsWith("for")) || node.code.contains("while")){  
            multi_or_single = true
          }
          else if(node.code.contains("if") || node.code.contains("else") || node.code.contains("switch") || node.code.contains("case") || node.code.contains("default")){
            multi_or_single = false
          }
          multi_or_single
        }

        p_branch = multi_Or_Single(p)
        q_branch = multi_Or_Single(q)

         

        def mutliFirst() : Unit = {
          var p_q_parent : AstNode = same_maxid_parent(p,q)
           
           
          if(p.code.contains("case") || p.code.contains("default")){
            if(p_q_parent.astParent.isControlStructure && p_q_parent.astParent.code.contains("switch")){
               
 
              var is_find : Boolean = false
              for(node <- find_some_node(p,q,"case")._2){
                if(node.astParent.astParent == p.astParent.astParent){
 
                  is_find = true
                }
              }
              for(node <- find_some_node(p,q,"default")._2){
                if(node.astParent.astParent == p.astParent.astParent){
                  is_find = true
                }
              }
              if(is_find){
                relation = 4
              }else{
                relation = 1 
              }
            }else{
              relation = 2  
            }
          }else if(p.code.contains("if") || p.code.contains("else")){
             
            if(is_father(p,q)){
               
               
              if(p.code.contains("if")){
                if(p.astChildren.l(1).ast.l.contains(q)){
                  relation = 1
                }else{
                   
                  relation = 4
                }
              }else if(p.code.contains("else")){
                 
                relation = 1  
              }
            }else{
              relation = 2
            }
          }
        }
        if(p_branch && q_branch){  
          if(is_father(p,q)){
            relation = 1
          }else{
            relation = 2
          }
        }
        else if(p_branch && !q_branch){   
          if(is_father(p,q)){
            relation = 1
          }else{
            relation = 2
            if(q.code.contains("if") || q.code.contains("if")){
              if(same_maxid_parent(p,q).isControlStructure && same_maxid_parent(p,q).code.contains("if")){
                 
                relation = 3
              }
            }else if(q.code.contains("case") || q.code.contains("default")){
               
              var p_q_parent: AstNode = same_maxid_parent(p,q)
              if(p_q_parent.isBlock && p_q_parent.astParent.code.contains("switch")){
                relation = 3
              }
            }
          }
        }
        else if(!p_branch && q_branch){  
          mutliFirst()
        }
        else if(!p_branch && !q_branch){  
           
          if((p.code.contains("if") || p.code.contains("else")) && (q.code.contains("if") || q.code.contains("else"))){
            relation = order_or_nest_if(p,q)
          }else if((p.code.contains("case") || p.code.contains("default")) && (q.code.contains("case") || q.code.contains("default"))){
            relation = order_or_nest_switch(p,q)
          }else if((p.code.contains("if") || p.code.contains("else")) && (q.code.contains("case") || q.code.contains("default"))){
            mutliFirst()
          }else if( (p.code.contains("case") || p.code.contains("default")) && (q.code.contains("if") || q.code.contains("else"))){
             
            mutliFirst()
          }
        }
        relation
      }

      def for_Belong_Which(p:AstNode,q:AstNode) : Int = {
        var for_Belong_Which : Int = 0
         
        var for_string :String = q.code.split("for").last.replace(" ", "")  
        var for_string_list = for_string.split(";")
        var for_string_hashmap = new mutable.HashMap[Int,Boolean]()  
        var num_true = 0
        var for_real = new mutable.HashMap[Int,Int]()  
        for(k <- 0 to for_string_list.length-1){
          if(for_string_list(k).length > 1){
            for_string_hashmap(k+1) = true  
            num_true += 1
          }else{
            for_string_hashmap(k+1) = false
          }
        }
         
        var num_real = 0
        for( k <- 0 to for_string_list.length-1 if(for_string_hashmap(k+1)==true)){
          num_real += 1
          for_real(k+1) = num_real  
        }
         
        breakable{
          for(m <- 0 to 2){
            if(for_string_hashmap(m+1) == true){
               
              if(q.astChildren.l(for_real(m+1)-1).ast.l.contains(p)){
                for_Belong_Which = m+1
                break
              }
            }
          }
        }
         
        for_Belong_Which
      }

      def find_outer_if(p:AstNode):AstNode = {
        var p_parent:AstNode = p.astParent
        while((p.astParent.code.contains("if"))||(p.astParent.code.contains("else"))){
           
          p_parent = p_parent.astParent
        }
        p_parent
      }
      def find_some_node(p:AstNode,q:AstNode,find_string:String):(Boolean,ListBuffer[AstNode]) = {
         
        var is_find :Boolean = false
        var find_node = new ListBuffer[AstNode]
        var q_father: AstNode = q.astParent
        var q_father_Previous_Round : AstNode = q

        while(q_father != same_maxid_parent(p,q).astParent){
          for(node <- q_father.ast.l if (node.id < q_father_Previous_Round.id && node.id > p.id)){
             
            if(node.code.contains(find_string)){
              if(node.isControlStructure){
                 
                if(find_string == "break" || find_string == "continue"){
                  is_find = true
                  find_node += node
                   
                }
              }
              else{ 
                is_find = true  
                find_node += node 
              }
            }
          }
          q_father_Previous_Round = q_father
          q_father = q_father.astParent
        }      
        (is_find,find_node)
      }

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

      def find_between_control(i:Int):ListBuffer[AstNode] = {
        var find_result = new ListBuffer[AstNode]
        var q_parent:AstNode = list_identifier_use(next_same_name(i)).astParent
         
         
 
 
        breakable{
          while(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
             
            if(q_parent.isControlStructure)
            {
              if(q_parent.code.contains("while")){
                if(q_parent.code.contains("do")){ 
                  find_result += q_parent
                  if(q_parent.astParent.id > cpg.method.l(1).id){
                    q_parent = q_parent.astParent
                  }else{
                    break
                  }
                }else{ 
                   
                  find_result += q_parent
                  if(q_parent.astParent.id > cpg.method.l(1).id){
                    q_parent = q_parent.astParent
                  }else{
                    break
                  }
                }
              }
              else if(q_parent.code.contains("for") && q_parent.code.startsWith("for")){
                find_result += q_parent
                if(q_parent.astParent.id > cpg.method.l(1).id){
                  q_parent = q_parent.astParent
                }else{
                  break
                }
              }
              else if(q_parent.code.contains("if")){
                 
                find_result += q_parent
                breakable{
                  while(q_parent.code.contains("if")){
                     
                     
                     
                    if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else") && q_parent.astParent.astChildren.l.length == 1){
                       
                      q_parent = q_parent.astParent.astParent.astParent  
                      break()
                    }else{
                       
                      q_parent = q_parent.astParent
                      break()
                    }
                     
                     
                     
                     
                     
                     
                  }
                }
              }
              else if(q_parent.code.contains("else")){
                 
                find_result += q_parent
                q_parent = q_parent.astParent  
                 
                if(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
                  breakable{
                    while(q_parent.code.contains("if")){
                      if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else")){
                         
                         
                         
                        if(q_parent.astParent.astParent.astChildren.astChildren.l.length == 1){
                           
                          q_parent = q_parent.astParent.astParent.astParent  
                          break()
                        }else{
                           
                          q_parent = q_parent.astParent
                           
                          break()
                        }
                      }else{
                        q_parent = q_parent.astParent
                        break()
                      }
                    }
                  }
                }
              }
              else if(q_parent.code.contains("switch")){ 
                 
                /*这种情况是
                  2 switch外到switch内的判断语句这种情况要排除（？？？）
                  3 switch外到switch内的分支语句
                 */
                 
                if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
                   
                   
                  find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
                }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
                   
                  find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
                }else{ 
                   
                  find_result += q_parent
                }
                 
                if(q_parent.astParent.id > cpg.method.l(1).id){
                  q_parent = q_parent.astParent
                }else{
                  break
                }
              }
            }
            else{
               
               
               
              if(q_parent.astParent.id > cpg.method.l(1).id){
                q_parent = q_parent.astParent
              }else{
                break
              }
               
               
            }
             
 
          }
        }


         
        if(q_parent == same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
           
          if(q_parent.astParent.code.contains("switch")){
             
             
             
            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
            }

          }else if(q_parent.isControlStructure && q_parent.code.contains("switch")){
             
             
             
             
            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
               
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
            }
          }
        }


         
         
         
         
         
         
        find_result
      }


      /*
      针对下一同名变量是普通变量的情况，由于有可能是入块的变量（普通变量也有可能是入块变量），并根据当前变量的类型有不同的处理
      函数对三种状态的判断是关于下一同名变量是否更新数据流的情况做的区分
      调用：common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1,next_is_out_control(0)._1,next_is_out_control(0)._2)
       */
      def common_yy_yn(i:Int,this_is_judge:String,this_is_out_control:String):Unit = {
         
         
        if(! update_data(i)._1){
          if(update_data(i)._2 == 2){
             
             
 
            if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")){
               
              draw_line(i,list_identifier_use(next_same_name(next_same_name(i))))
              for (control_node <- 0 to find_between_control(i).length - 1) {
                if(map_newuse.get(list_identifier_use(i).code) == None){
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }else{
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }
              }
            }
            else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
               
              var list_out_data = new ListBuffer[AstNode]
              if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && !map_last_stack(list_identifier_use(i).code).isEmpty){
                for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                  list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
                  builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                }
                for (control_node <- 0 to find_between_control(i).length - 1) {
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }
                map_last_stack(list_identifier_use(i).code).pop()
              }
            }
            update_map_newuse(i,list_identifier_use(next_same_name(next_same_name(i))))
          }
          else{
 
             
 
            if(update_data(i)._2 == 3){  
              builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
            }
            if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")) {
 
               
              draw_line(i, list_identifier_use(next_same_name(i)))
 
 
              for (control_node <- 0 to find_between_control(i).length - 1) {
 
                if(map_newuse.get(list_identifier_use(i).code) == None){
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }else{
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
            }
            else if ((this_is_judge == "false" && this_is_out_control != "false") || (this_is_judge != "false" && this_is_out_control != "false")) {
              var list_out_data = new ListBuffer[AstNode]


              if(!map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && !map_last_stack(list_identifier_use(i).code).isEmpty){
 
                 

                for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                  list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
                  builder.addEdge(list_identifier_use(next_same_name(i)),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                }

                for (control_node <- 0 to find_between_control(i).length - 1 ) {
 
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
              if(!map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && !map_last_stack(list_identifier_use(i).code).isEmpty){
                map_last_stack(list_identifier_use(i).code).pop()
              }
            }
            update_map_newuse(i,list_identifier_use(next_same_name(i)))
          }
        }
        else{
 
           
          if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")) {
 
 
            for (control_node <- 0 to find_between_control(i).length - 1 ) {
               
              if(map_newuse.get(list_identifier_use(i).code) == None){
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }else{
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }
            }
          }
          else if ((this_is_judge == "false" && this_is_out_control != "false") || (this_is_judge != "false" && this_is_out_control != "false")) {
            var list_out_data = new ListBuffer[AstNode]
            if(!map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && ! map_last_stack(list_identifier_use(i).code).isEmpty){
 
              for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
              }

              for (control_node <- 0 to find_between_control(i).length - 1 ) {
 
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  if(next_same_name(next_same_name(i)) != -1){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }else{
                  if(next_same_name(next_same_name(i)) != -1){
 
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }
              }
              map_last_stack(list_identifier_use(i).code).pop()
            }
          }
          update_map_newuse(i,list_identifier_use(next_same_name(i)))
        }
      }


      /*
        针对下一同名变量是判断
       */
      def for_Inner(i:Int,next_is_out_control:String,next_is_out_control_node:AstNode):Unit = {
        var is_inner:Boolean = false
        if(next_is_out_control != "false"){ 
          if(next_is_out_control_node.code.contains("for") && next_is_out_control_node.code.startsWith("for")){ 
            if(map_For_Inner.get(next_is_out_control_node) != None){ 
              if(map_For_Inner(next_is_out_control_node).contains(list_identifier_use(i).code)){
                is_inner = true
              }
            }
            if(is_inner){ 
              if(map_For_Outer.get(next_is_out_control_node) != None){
                if(map_For_Outer(next_is_out_control_node).get(list_identifier_use(i).code) != None){
                  for(k <- 0 to map_For_Outer(next_is_out_control_node)(list_identifier_use(i).code).length-1){
 
                    builder.addEdge(map_For_Outer(next_is_out_control_node)(list_identifier_use(i).code)(k),list_identifier_use(next_same_name(i)),"LastUse")
                  }
                }else{
                  if(map_For_Second.get(next_is_out_control_node) != None){
                    if(map_For_Second(next_is_out_control_node).get(list_identifier_use(i).code) != None){
                      for(k <- 0 to map_For_Second(next_is_out_control_node)(list_identifier_use(i).code).length-1){
                        builder.addEdge(map_For_Second(next_is_out_control_node)(list_identifier_use(i).code)(k),list_identifier_use(next_same_name(i)),"LastUse")
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      /*
      针对下一同名变量是入块变量的情况，根据当前变量的情况分情况讨论
      调用：common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
       */
      def common_ny_nn(i:Int, this_is_judge:String, this_is_out_control:String):Unit = {
         
        if(this_is_judge == "false" && this_is_out_control == "false"){
           
          draw_line(i, list_identifier_use(next_same_name(i)))
 
 
           
          for (control_node <- 0 to find_between_control(i).length - 1) {
             
            if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for") && find_between_control(i)(find_between_control(i).length - 1 - control_node).code.startsWith("for")){
              var for_belong_which = for_Belong_Which(list_identifier_use(next_same_name(i)),find_between_control(i)(find_between_control(i).length - 1 - control_node))
              if(for_belong_which == 1){
                if(map_For_First.get(find_between_control(i)(find_between_control(i).length - 1 - control_node)) != None ){
                  if(map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                    map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                  }
                  else{
                    map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                  }
                }else{
                  map_For_First.put(find_between_control(i)(find_between_control(i).length - 1 - control_node),mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                   
                }
                 
                 
                draw_line(i,list_identifier_use(next_same_name(i)))
              }else if(for_belong_which == 2 || for_belong_which == 3){
                if(for_belong_which == 2){
 
                  if(map_For_Second.get(find_between_control(i)(find_between_control(i).length - 1 - control_node)) != None ){
                    if(map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                      map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                    }
                    else{
                      map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                    }
                  }else{
                    map_For_Second.put(find_between_control(i)(find_between_control(i).length - 1 - control_node),mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }else if(for_belong_which == 3){
 
                  if(map_For_Outer.get(find_between_control(i)(find_between_control(i).length - 1 - control_node)) != None ){
                    if(map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                      map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                    }
                    else{
                      map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                    }
                  }else{
                    map_For_Outer.put(find_between_control(i)(find_between_control(i).length - 1 - control_node),mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
 
                }
                 
                draw_line(i,list_identifier_use(next_same_name(i)))
                if(map_newuse.get(list_identifier_use(i).code) == None){
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }else{
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
            }
            else{  
              if(map_newuse.get(list_identifier_use(i).code) == None){
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }else{
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(map_newuse(list_identifier_use(i).code).head), ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }
            }
          }
          update_map_newuse(i,list_identifier_use(next_same_name(i)))
        }
        else if((this_is_judge == "false" && this_is_out_control != "false") || (this_is_judge != "false" && this_is_out_control != "false")){
           
          var list_out_data = new ListBuffer[AstNode]
          if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && ! map_last_stack(list_identifier_use(i).code).isEmpty){
            for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
              list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
              builder.addEdge(list_identifier_use(next_same_name(i)),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
            }
            for (control_node <- 0 to find_between_control(i).length - 1) {
               
              if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for") && find_between_control(i)(find_between_control(i).length - 1 - control_node).code.startsWith("for")){
                var for_belong_which = for_Belong_Which(list_identifier_use(next_same_name(i)),find_between_control(i)(find_between_control(i).length - 1 - control_node))
                if(for_belong_which == 1){
                   
                   
                  draw_line(i,list_identifier_use(next_same_name(i)))
                }else if(for_belong_which == 2 || for_belong_which == 3){
                   
                  draw_line(i,list_identifier_use(next_same_name(i)))
                  if(map_newuse_stack.isEmpty || map_newuse_stack.get(list_identifier_use(i).code) == None || map_newuse_stack(list_identifier_use(i).code).isEmpty){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                    if(next_same_name(i) != -1 && next_same_name(next_same_name(i)) != -1){
                      map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                    }
                  }
                }
              }
              else{ 
                if(map_newuse_stack.isEmpty || map_newuse_stack.get(list_identifier_use(i).code) == None || map_newuse_stack(list_identifier_use(i).code).isEmpty){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
 
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }
            }
            map_last_stack(list_identifier_use(i).code).pop()
          }
          update_map_newuse(i,list_identifier_use(next_same_name(i)))
        }
      }

      /*
      针对下一同名变量是普通变量的情况，由于有可能是入块的变量，并根据当前变量的类型有不同的处理
      调用：if_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
       */
      def if_yy_yn(i:Int,this_is_judge:String,this_is_judge_node:AstNode,this_is_out_control:String):Unit = {
         
        if(! update_data(i)._1){  
          if(update_data(i)._2 == 2){ 
            if(this_is_judge != "false" && this_is_out_control == "false"){ 
              /*
              当前变量为判断语句，下一同名变量为普通变量的情况：（while和for应该是只有这种情况（即当前变量是判断语句））
              1 将下一同名变量与判断语句中所有的同名变量连边
              2 判断为普通变量的下一同名变量是否为其他块的入块变量（使用find_between_control），若有，入栈、连边
               */
               
               
               
              if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                 
                if(this_is_judge_node.code.contains("for") && this_is_judge_node.code.startsWith("for")){
                  var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),this_is_judge_node)
                  /*
                  如果for_Belong_This == 1 ，代表只有第一个分支，那么就由第一个分支连边
                  如果for_Belong_This == 2 ，代表第二个分支存在，只要第二个分支存在就使用第二个分支的变量连接（即使第一个分支也存在）
                  如果for_Belong_This == 3 ，代表第三个分支存在，无论前两个分支存不存在，都需要下一同名变量连向第三个分支，
                      但是需要判断，如果第二个分支存在（也可以是一二分支都存在），那么下一同名变量需要连向第二个分支的变量，
                                  如果第二个分支不存在，判断第一个分支是否存在，如果存在（一存在，二不存在），增加连接向第一个分支的边
                  所以无论for_Belong_This的取值，都需要下一同名变量向上连边
                   */
                  if(for_Belong_This == 3){
                    if(map_For_Second.get(this_is_judge_node)!= None && map_For_Second(this_is_judge_node).get(list_identifier_use(i).code)!= None){
                      for(k <- 0 to map_For_Second(this_is_judge_node)(list_identifier_use(i).code).length - 1){
                        builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),map_For_Second(this_is_judge_node)(list_identifier_use(i).code)(k),"LastUse")
                      }
                    }else{
                      if(map_For_First.get(this_is_judge_node)!= None && map_For_First(this_is_judge_node).get(list_identifier_use(i).code) != None){
                        for(k <- 0 to map_For_First(this_is_judge_node)(list_identifier_use(i).code).length - 1){
                          builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),map_For_First(this_is_judge_node)(list_identifier_use(i).code)(k),"LastUse")
                        }
                      }
                    }
                  }
                }
                else{  
                  for(node <- map_newuse_stack(list_identifier_use(i).code).head._3){
                    builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),node,"LastUse")
                  }
                }
              }else{  
                builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),list_identifier_use(i),"LastUse")
              }
               
              for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
                 
                 
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                }else{ 
                  /*
                  也需要分情况讨论：
                  若当前map_newuse_stack栈顶保存的._1块节点不是当前变量所在的块节点，说明栈顶保存的是上一轮for循环中，内部其他块也将下一同名变量作为入块变量的情况
                  */
                  if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_judge_node){
                     
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                     
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._2
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }
              }
            }
            else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
               
              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                 
                 
                if(find_between_control(i).length > 0){
                  if (order_or_nest_plus(map_newuse_stack(list_identifier_use(i).code).head._1, find_between_control(i).last) == 5) {
                    for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
                      map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                    }
                  }
                   
                  for (p <- 0 to list_data.length - 1) {
                    builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), list_data(p), "LastUse")
                  }
                }else{
                  draw_line(i,list_identifier_use(next_same_name(next_same_name(i))))
                }
              }
            }
             
            update_map_newuse(i,list_identifier_use(next_same_name(next_same_name(i))))
          }
          else{
            if(update_data(i)._2 == 3){  
              builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
            }
            if(this_is_judge != "false" && this_is_out_control == "false"){
               
              if(find_between_control(i).length > 1){
                for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
                  if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for") && find_between_control(i)(find_between_control(i).length - 1 - control_node).code.startsWith("for")){
                     
                    var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),find_between_control(i)(find_between_control(i).length - 1 - control_node))
                    if(for_Belong_This == 3){
                       
                      if(map_For_Second.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                         
                        for(k <- 0 to map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                          builder.addEdge(list_identifier_use(next_same_name(i)),map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                        }
                      }else{
                         
                        if(map_For_First.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                           
                          for(k <- 0 to map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                            builder.addEdge(list_identifier_use(next_same_name(i)),map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                          }
                        }
                      }
                       
                       
                      for(k <- 0 to map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                        builder.addEdge(list_identifier_use(next_same_name(i)),map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                      }
                    }
                    else{
                      builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(i),"LastUse")
                    }
                  }
                   
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{ 
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
              else{
 
                 
                 
 
                if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                  for(line <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse_stack(list_identifier_use(i).code).head._3(line),"LastUse")
                  }
                }
              }

               
              update_map_newuse(i,list_identifier_use(next_same_name(i)))
            }
            else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                 
                if(find_between_control(i).length > 0){
                  if (order_or_nest_plus(map_newuse_stack(list_identifier_use(i).code).head._1, find_between_control(i).last) == 5) {
                     
                    map_newuse_stack(list_identifier_use(i).code).pop()
                     
                    for (control_node <- 0 to find_between_control(i).length - 1) {
                      map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                    }
                  }
                   
                  for (p <- 0 to list_data.length - 1) {  
                    builder.addEdge(list_identifier_use(next_same_name(i)), list_data(p), "LastUse")
                  }
                }
                else{
                   
                  draw_line(i,list_identifier_use(next_same_name(i)))
                }
              }
            }
            update_map_newuse(i,list_identifier_use(next_same_name(i)))
             
          }
        }
        else{
          if(this_is_judge != "false" && this_is_out_control == "false"){
             
             
            for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
              if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
              }else{ 
                var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(i)))))
              }
            }
          }
          else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
             
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {

 
              if(find_between_control(i).length > 0){
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                 
                if (order_or_nest_plus(map_newuse_stack(list_identifier_use(i).code).head._1, find_between_control(i).last) == 5) {
                  for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
            }
          }
          update_map_newuse(i,list_identifier_use(next_same_name(i)))
        }
      }


      /*
      针对下一同名变量是普通变量的情况，由于有可能是入块的变量，并根据当前变量的类型有不同的处理
      调用：if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
       */
      def if_ny_nn(i:Int,this_is_judge:(String, AstNode),next_is_judge:(String, AstNode),this_is_out_control:List[(String,AstNode)]):Unit = {
        if(this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false"){
           
          if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
            var newuse_3: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._3  
             
             
            for (p <- 0 to newuse_3.length - 1) {
              builder.addEdge(list_identifier_use(next_same_name(i)), newuse_3(p), "LastUse")
            }
             
            for (q <- 0 to find_between_control(i).length - 1) { 
 

               
              if(order_or_nest_plus(this_is_judge._2,find_between_control(i)(find_between_control(i).length - 1 - q)) == 1){
                map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - q), newuse_3, ListBuffer(list_identifier_use(next_same_name(i)))))

              }
 
 
 
            }
          }
        }
        else if((this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false")||(this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false")){
 
          if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty){
 

             

 
            var newuse_2:ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2 
             
            map_newuse_stack(list_identifier_use(i).code).pop()
 
            /*
              next_is_judge有3种情况：并且这三种情况this_is_out_control.last._2的id小于next_is_judge
              1 next_is_judge是嵌套在this_is_out_control.last._2内的块
              不存在该种情况，因为当前变量已经是出结构体的变量了
              2 next_is_judge是this_is_out_control.last._2的下行顺序if分支
              3 next_is_judge是this_is_out_control.last._2的下行顺序if分支中嵌套的块
             */

             
 
 
 
 
 
            if(order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2) == 5){
               
               
               
              map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
               
              for(p <- 0 to newuse_2.length-1){
                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
              }
            }
            else if(order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2) == 4){
               
               
               
               
              for(p <- 0 to newuse_2.length-1){
                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
              }
               
 
              for(q <- 0 to find_between_control(i).length-1) {
                 
                 
 
 
                if(find_between_control(i)(find_between_control(i).length-1-q) == next_is_judge._2){
 
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length-1-q),newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
                }
 
                if(order_or_nest_plus(find_between_control(i)(find_between_control(i).length-1-q),next_is_judge._2) == 1) { 
 
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length-1-q),newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }
            }
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
          }
        }
        update_map_newuse(i,list_identifier_use(next_same_name(i)))
      }

      /*
       主要处理当前变量为出if某个分支的操作，
       表现为：两个变量公共父节点是控制节点if，主要是if -5678情况
       处理：因为当前变量为出结构体变量，可能为同时出多个结构体，即多个结构体嵌套的情况
            内层嵌套为while主要考虑的操作：出数据流与其判断语句中的变量（是否有也需要判断，通过map_newuse_stack）的连边；
            内层嵌套为if主要考虑的操作：当前if分支的出数据流在

       最外层肯定出的是if分支，虽然有可能为一个变量同时出多个块
       对于出多分支的某个分支，需要用map_last_stack记录该分支的出数据流作为整个多分支出数据流的一部分
       */
      def out_control(i:Int,this_is_out_control:List[(String,AstNode)]):Unit = {
 
           
        for (node <- 0 to this_is_out_control.length-1){
 
           
           
          if(this_is_out_control(node)._1 == "while"){
 
             
             
            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                 
                /*如果当前map_lastuse_stack保存的栈顶节点是while节点的子节点，说明while内部有其他多分支的块，
                  即出数据流不仅只有当前变量，还有while内部多分支的块，通过看while节点与map_last_stack保存的控制节点的关系，
                  如果是父节点，那么就说明确实存在别的块，需要考虑连边，
                  连边之后map_last_stack需要继续保存，属于当前while的map_newuse_stack栈顶的元素需要弹出
                 */
                if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && ! map_last_stack(list_identifier_use(i).code).isEmpty){
                  if(is_father(this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1)){
                     
                    for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                      for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                        builder.addEdge(map_last_stack(list_identifier_use(i).code).head(k)._2,map_newuse_stack(list_identifier_use(i).code).head._3(m),"LastUse")
                      }
                    }
                  }
                }
                else{
                  for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    if(map_newuse.get(list_identifier_use(i).code) == None){
                      builder.addEdge(list_identifier_use(i),map_newuse_stack(list_identifier_use(i).code).head._3(m),"LastUse")
                    }else{
                      builder.addEdge(map_newuse(list_identifier_use(i).code).head,map_newuse_stack(list_identifier_use(i).code).head._3(m),"LastUse")
                    }
                  }
                }
              }
               
              map_newuse_stack(list_identifier_use(i).code).pop()
            }
          }
          else if(this_is_out_control(node)._1 == "if" || this_is_out_control(node)._1 == "switch"){
             
             
             
 

            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
               
              map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
            }
            else{
              /*
              分为两种情况
              1 当前出的if是内层if，并且肯定该内层if也是完全出结构体，所以需要将栈顶保存的内层if的出数据流合并到外层（可能是多种块）的出数据流里
              2 当前出的if是出的最外层的整个多分支if，但是也分为两种情况
                2.1 该外层if分支的内部有其他出块，所以当前map_last_stack栈顶保存的是内部出块的出数据流，(map_last_stack栈顶保存的节点id大于当前出分支节点id)
                2.2 该外层if分支的内部没有其他出块，直接将当前出分支数据流并入map_last_stack栈顶，()
               */
              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
 
                 
                var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
                map_last_stack(list_identifier_use(i).code).pop()
                if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
                   
                   
                  var list_change_name = new ListBuffer[(AstNode,AstNode)]
                  for (list_node <- 0 to list_tmp.length-1){
                    list_change_name +=((this_is_out_control(node)._2,list_tmp(list_node)._2))
                  }
                  map_last_stack(list_identifier_use(i).code) = Stack(list_change_name)
                }
                else{
                  if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                     
                    var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                    map_last_stack(list_identifier_use(i).code).pop()
                    map_last_stack(list_identifier_use(i).code).push(list_concat)
                  }else{
                     
                    map_last_stack(list_identifier_use(i).code).push(list_tmp)
                  }
                }
              }
              else{
 
                var list_tmp = new ListBuffer[(AstNode,AstNode)]
                if(map_newuse.get(list_identifier_use(i).code) != None){
                  list_tmp= ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head))
                }
                else{
                  list_tmp = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i)))
                }
                 
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                   
                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)
                }
                else{
                   
                  map_last_stack(list_identifier_use(i).code).push(list_tmp)
                }
              }
            }


            /*
            由于当前变量是出块变量，且是内层的嵌套的出块变量，因为无论当前块的判断语句中是否有同名变量（因为即使是普通变量也有可能是入块变量），
            所以要在出内层块的时候，将map_newuse_stack栈顶弹出栈
            即使是最差的情况，块中只有一个同名变量，当时肯定也使用该变量为当前块入map_newuse_stack栈

            内层的出if为出整个if多分支，需要将map_newuse_stack栈顶弹出，最外层的if只是出当前分支，所以暂时不弹出，将入数据流交接完再弹出
             */
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None &&  ! map_newuse_stack(list_identifier_use(i).code).isEmpty && node != this_is_out_control.length-1){
              map_newuse_stack(list_identifier_use(i).code).pop()
            }
            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
          }
          else if(this_is_out_control(node)._1 == "for"){
             

          }
          else if(this_is_out_control(node)._1 == "do-while"){

          }
        }
      }

      /*
      处理当前变量为普通变量且为出整个块的情况yn和nn
      应该和out_control的思路不一样，for循环应该直接把所有出的层处理掉
       */
      def out_control_common(i:Int,this_is_out_control:List[(String,AstNode)]):Unit = {
        for (node <- 0 to this_is_out_control.length-1){
 
           
           
          if(this_is_out_control(node)._1 == "while"){
             
 
             
             
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
               
              if(map_newuse.get(list_identifier_use(i).code) != None){
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
              }else{
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
              }
            }
            else{
              /*
              map_last_stack不为空：
              有两种情况（可能当前栈顶不是当前判断的出块的节点）
              1 栈顶保存的是while内部嵌套的其他块的出变量，有可能是多分支的情况，需要将栈顶合并进当前while块的出数据流（即while中嵌套了其他完整的块）
                while(){for(){x}} x为同时出for和while的变量，此时map_last_stack栈顶保存的是for节点
                判断：此时map_last_stack栈顶的节点的id > 当前出块变量的id
              2 栈顶保存的是其他块的出分支变量，并且当前块是多分支块，还没有完全出块（即while被嵌套在多分支中）
                if(){x}else if(){while(){x}} 此时map_last_stack栈顶保存的就是if第一个分支的出数据流
                判断：此时map_last_stack栈顶的节点的id < 当前出块变量的id
               */
              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
                 
                if(order_or_nest_plus(this_is_out_control(node)._2, map_last_stack(list_identifier_use(i).code).head(0)._1) == 1){
                   
                  var list_tmp:ListBuffer[(AstNode, AstNode)] = map_last_stack(list_identifier_use(i).code).head
                  var list_concat = new ListBuffer[(AstNode, AstNode)]
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    list_concat = ListBuffer((this_is_out_control(node)._2, map_newuse(list_identifier_use(i).code).head)) ++ list_tmp
                  }else{
                    list_concat = ListBuffer((this_is_out_control(node)._2, list_identifier_use(i))) ++ list_tmp
                  }
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)
                }
              }
              else if(map_last_stack(list_identifier_use(i).code).head(0)._1.id < this_is_out_control(node)._2.id){
                 
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 4){
                   
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
                  }else{
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
                  }
                }
              }
            }

             
             
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){

 
               
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                 
                 
                if(map_last_stack.get(list_identifier_use(i).code) != None){
                   
                  if(this_is_out_control(node)._2 == map_last_stack(list_identifier_use(i).code).head(0)._1){
                     
                    for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                      for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                        builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m), map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                      }
                    }
                  }
                }
                else{
                   
                  for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    if(map_newuse.get(list_identifier_use(i).code) == None){
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),list_identifier_use(i),"LastUse")
                    }else{
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }
                  }
                }
                 
                map_newuse_stack(list_identifier_use(i).code).pop()
              }
            }
            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
          }
          else if(this_is_out_control(node)._1 == "if" || this_is_out_control(node)._1 == "switch"){
 
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
               
              map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
            }
            else
            {
              /*
              分为两种情况
              1 当前出的if是内层if，并且肯定该内层if也是完全出结构体，所以需要将栈顶保存的内层if的出数据流合并到外层（可能是多种块）的出数据流里
              2 当前出的if是出的最外层的整个多分支if，但是也分为两种情况
                2.1 该外层if分支的内部有其他出块，所以当前map_last_stack栈顶保存的是内部出块的出数据流，(map_last_stack栈顶保存的节点id大于当前出分支节点id)
                2.2 该外层if分支的内部没有其他出块，直接将当前出分支数据流并入map_last_stack栈顶，()
               */
              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
 
                 
                var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
                map_last_stack(list_identifier_use(i).code).pop()
                if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
                   
                   
                  var list_change_name = new ListBuffer[(AstNode,AstNode)]
                  for (list_node <- 0 to list_tmp.length-1){
                    list_change_name +=((this_is_out_control(node)._2,list_tmp(list_node)._2))
                  }
                  map_last_stack(list_identifier_use(i).code) = Stack(list_change_name)
                }
                else{
                  if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                     
                    var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                    map_last_stack(list_identifier_use(i).code).pop()
                    map_last_stack(list_identifier_use(i).code).push(list_concat)
                  }else{
                     
                    map_last_stack(list_identifier_use(i).code).push(list_tmp)
                  }
                }
              }
              else{
 
                var list_tmp = new ListBuffer[(AstNode,AstNode)]
                if(map_newuse.get(list_identifier_use(i).code) != None){
                  list_tmp= ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head))
                }
                else{
                  list_tmp = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i)))
                }
                 
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                   
                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)
                }
                else{
                   
                  map_last_stack(list_identifier_use(i).code).push(list_tmp)
                }
              }
            }

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
             
             
             

 
 
 
 
 
 
 
 
 
 
 
 
            if(!map_newuse_stack.isEmpty){
              if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                val containsEmptyTuple: Boolean = map_newuse_stack(list_identifier_use(i).code).headOption.exists(_ == (()))

                if(!containsEmptyTuple){
 
 
                  if(map_last_stack(list_identifier_use(i).code).isEmpty){
 
                    map_newuse_stack(list_identifier_use(i).code).pop()
                  }


                }
              }

            }

 
 
 
 

            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
          }
          else if(this_is_out_control(node)._1 == "for"){

          }
          else if(this_is_out_control(node)._1 == "do-while"){
             
 
            /*
            while的判断语句和出do块（while中没有同名变量）作为出结构体的变量，都会进入这里
            需要做的操作：
            1 将出块变量入map_last_stack栈，和while一样，也要根据map_last_stack栈顶保存的元素分情况讨论
            2 和while的情况一样，因为是个循环体，出块变量会影响到入块变量，（while语句出块和do块普通变量作为出块是一样的）
             */
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
               
              if(map_newuse.get(list_identifier_use(i).code) != None){
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
              }else{
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
              }
            }
            else{
               

              /*
              不为空，可能当前栈顶不是当前判断的出块的节点
              有两种情况，与while的情况完全相同
              1 栈顶保存的是while内部嵌套的其他块的出变量，有可能是多分支的情况，需要将栈顶合并进当前while块的出数据流
              2 栈顶保存的是其他块的出块变量，并且当前块是多分支块，还没有完全出块
                if(){x}else if(){while(){x}} 此时map_last_stack栈顶保存的就是if第一个分支的出数据流
               */

               
              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
                if(order_or_nest_plus(this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1) == 1){
                   
                  var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
                  var list_concat = new ListBuffer[(AstNode,AstNode)]
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    list_concat = ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)) ++ list_tmp
                  }else{
                    list_concat = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))) ++ list_tmp
                  }
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)

                }
              }
              else if(map_last_stack(list_identifier_use(i).code).head(0)._1.id < this_is_out_control(node)._2.id){
                 
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 4){
                   
                   
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
                  }else{
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
                  }
                }
              }
            }

            if(map_newuse_stack.get(list_identifier_use(i).code) != None){
               
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                 
                 
                if(map_last_stack.get(list_identifier_use(i).code) != None){
                   
                  if(this_is_out_control(node)._2 == map_last_stack(list_identifier_use(i).code).head(0)._1){
                     
                    for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                      for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
 
                        builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                      }
                    }
                  }
                }
                else{
                   
                  for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    if(map_newuse.get(list_identifier_use(i).code) == None){
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),list_identifier_use(i),"LastUse")
                    }else{
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }
                  }
                }
                 
                map_newuse_stack(list_identifier_use(i).code).pop()
              }
            }

            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
 
            }else{
 
            }
          }
 
 
 
        }
 

      }


       
       
      for (i <- 0 to list_identifier_use.length-1) {
        breakable {
           
          if (i == list_identifier_use.length - 1) {
            break
          }
          else if (next_same_name(i) == -1) {  
             
 
            break
          }

           
           
          var same_parent_maxid: AstNode = list_identifier_use(i)
          if (next_same_name(i) != -1) {
            same_parent_maxid = same_maxid_parent(list_identifier_use(i), list_identifier_use(next_same_name(i)))
 
          }

           
          var this_is_judge: (String, AstNode) = is_judge(i)
          var next_is_judge: (String, AstNode) = is_judge(next_same_name(i))
          var this_is_out_control = is_out_control(i).toList.map { case (key, value) => (value, key) }
          var next_is_out_control = is_out_control(next_same_name(i)).toList.map { case (key, value) => (value, key) }
 

          if(map_newuse_stack.get(list_identifier_use(i).code) != None){
 
          }else{
 
          }
          if(map_last_stack.get(list_identifier_use(i).code) != None){
 
          }else{
 
          }

          if (same_parent_maxid.code.contains("<empty>") || same_parent_maxid == list_identifier_use(i))
          {
            if(same_parent_maxid.astParent.isControlStructure && same_parent_maxid.astParent.code.contains("switch"))
            {
               
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
 
                 
                out_control(i,this_is_out_control)
                 
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
 
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                 
 
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                 
                 
 
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
            }
            else
            {
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                 
 
                 
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
                 
                 
                 
                 
 
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
                 
                for_Inner(i,next_is_out_control(0)._1,next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
                 
 
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                 
 
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if ((this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false")
                ||(this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false")) {
                 
                 
 
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
                for_Inner(i,next_is_out_control(0)._1,next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
 
                out_control_common(i,this_is_out_control)
                 
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)

                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
 
                out_control_common(i,this_is_out_control)
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)

                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                 
                 
 
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
 
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false")  {
 
                 
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                 
 
              }

              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                 
 
                out_control_common(i, this_is_out_control)
                common_yy_yn(i, this_is_judge._1, this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
                 
 
                out_control_common(i, this_is_out_control)
                common_yy_yn(i,this_is_judge._1, this_is_out_control(0)._1)
                for_Inner(i,next_is_out_control(0)._1, next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
                out_control_common(i, this_is_out_control)
                common_ny_nn(i, this_is_judge._1, this_is_out_control(0)._1)

              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
 
                out_control_common(i,this_is_out_control)
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
            }
          }
          else if (same_parent_maxid.code.contains("while"))
          {
             
             
             
            if(same_parent_maxid.code.contains("do")){
               
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
 
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
 
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
 
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
            }
            else{  
              if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                 
                 
 
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                 
 
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
                 
 
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                 
 
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
            }

          }
          else if (same_parent_maxid.isControlStructure && same_parent_maxid.code.contains("for") && same_parent_maxid.isControlStructure && same_parent_maxid.code.startsWith("for"))
          {
            var list_In_Decrement = List("preIncrement","postIncrement","preDecrement","postDecrement")
            /*
            进入for的块有三种情况：
            1 for的前三个子分支都没有（即相当于while循环的情况 for(;;)）
              则
            2 for的第二个分支没有，即for(i=0;;i++),
              则
            3 for的三个分支都有
             */
             
             
             
 
            if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
              var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),this_is_judge._2)
              var for_Belong_Next: Int = for_Belong_Which(list_identifier_use(next_same_name(i)),next_is_judge._2)
               
              if(map_For_Inner.get(same_parent_maxid) != None && !map_For_Inner(same_parent_maxid).isEmpty)
              {
                 
                if(!map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code)){
                   
                  if(last_same_name(i) == -1){
                     
                    map_For_Inner(same_parent_maxid) += list_identifier_use(i).code
                     
                     
                  }
                }
              }
              else
              { 
 
                if(last_same_name(i) == -1){
                  map_For_Inner(same_parent_maxid) = ListBuffer(list_identifier_use(i).code)
                   
                }
              }
 

               
              if(for_Belong_Next == 4){
                if(for_Belong_This == 1){

                }
                else if(for_Belong_This == 2){

                }else if(for_Belong_This == 3){

                }
              }else{
                if(for_Belong_This == 1 && for_Belong_Next == 2)
                {
 
                  if(! map_For_Inner.isEmpty && map_For_Inner.get(same_parent_maxid) != None && ! map_For_Inner(same_parent_maxid).isEmpty){
                    if(map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code))
                    {
 
                      if(map_For_Record.get(same_parent_maxid) == None){
                        map_For_Record(same_parent_maxid) = ListBuffer(true,true)
                      }else{
                        map_For_Record(same_parent_maxid)(0) = true
                        map_For_Record(same_parent_maxid)(1) = true
                      }
                      val hashMap2 = new mutable.HashMap[String, ListBuffer[AstNode]]
                      hashMap2(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                      map_For_Second(same_parent_maxid) = hashMap2
 
                       
                       
                      builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(i),"LastUse")
                       
                      if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                        map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }else{
                        map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }
                    }
                  }
                  else
                  {
 
                     
                     
                     
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                     
                    if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                      map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }else{
                      map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }
                  }
                }
                else if(for_Belong_This == 1 && for_Belong_Next == 3)
                {
                   
                  if(! map_For_Inner.isEmpty && map_For_Inner.get(same_parent_maxid) != None && ! map_For_Inner(same_parent_maxid).isEmpty){
                    if(map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code))
                    {
 
                      if(map_For_Record.get(same_parent_maxid) == None){
                        map_For_Record(same_parent_maxid) = ListBuffer(true,false,true)
                      }else{
                        map_For_Record(same_parent_maxid)(0) = true
                        map_For_Record(same_parent_maxid)(1) = false
                        map_For_Record(same_parent_maxid)(2) = true
                      }

                       
                      if(map_For_First.get(same_parent_maxid) != None ){
                        if(map_For_First(same_parent_maxid).get(list_identifier_use(i).code) != None){
                          map_For_First(same_parent_maxid)(list_identifier_use(i).code) += list_identifier_use(i)
                        }
                        else{
                          map_For_First(same_parent_maxid)(list_identifier_use(i).code) = ListBuffer(list_identifier_use(i))
                        }
                      }else{
                        map_For_First.put(same_parent_maxid,mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(i))))
                      }
                       
                      if(map_For_Outer.get(same_parent_maxid) != None ){
                        if(map_For_Outer(same_parent_maxid).get(list_identifier_use(i).code) != None){
                          map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                        }
                        else{
                          map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                        }
                      }else{
                        map_For_Outer.put(same_parent_maxid,mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                      }

                      for(k <- 0 to list_In_Decrement.length-1){
                         
                        if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                          builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                        }
                      }

                       
                      if(map_newuse.get(list_identifier_use(i).code) != None){
                        map_newuse(list_identifier_use(i).code).push(list_identifier_use(i))
                      }else{
                        map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(i))
                      }
                    }
                  }
                  else
                  { 
 
                     
                    if(! map_newuse.isEmpty && map_newuse.get(list_identifier_use(i).code) != None && ! map_newuse(list_identifier_use(i).code).isEmpty){
                      builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }

                     
                    if(map_For_Outer.get(same_parent_maxid) != None ){
                      if(map_For_Outer(same_parent_maxid).get(list_identifier_use(i).code) != None){
                        map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                      }
                      else{
                        map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                      }
                    }else{
                      map_For_Outer.put(same_parent_maxid,mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                    }
                  }
                }
                else if(for_Belong_This == 2 && for_Belong_Next == 3)
                {
                  if(! map_For_Inner.isEmpty && map_For_Inner.get(same_parent_maxid) != None && ! map_For_Inner(same_parent_maxid).isEmpty){
                    if(map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code))
                    {
 
                      if(map_For_Record.get(same_parent_maxid) == None){
                         
                        map_For_Record(same_parent_maxid) = ListBuffer(false,true,true)
                      }else{
                         
                        map_For_Record(same_parent_maxid) += true
                      }
                      if(map_For_Outer.get(same_parent_maxid) != None ){
                        if(map_For_Outer(same_parent_maxid).get(list_identifier_use(i).code) != None){
                          map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                        }
                        else{
                          map_For_Outer(same_parent_maxid)(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                        }
                      }else{
                        map_For_Outer.put(same_parent_maxid,mutable.HashMap(list_identifier_use(i).code -> ListBuffer(list_identifier_use(next_same_name(i)))))
                      }

                      for(k <- 0 to list_In_Decrement.length-1){
                         
                        if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                          builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                        }
                      }

                      if(map_For_Record.get(same_parent_maxid) != None){
                        if(map_For_Record(same_parent_maxid)(0) == false){
                           
                           
                          if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                            map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                          }else{
                            map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                          }
                        }
                         
                        if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                          for(k2 <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                            builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse_stack(list_identifier_use(i).code).head._3(k2),"LastUse")
                          }
                        }
                      }

                       
                       
                       
                       
                       
                       
                    }
                  }
                  else
                  { 
 
                     
                    if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                       
                      builder.addEdge(list_identifier_use(i),map_newuse(list_identifier_use(i).code).head,"LastUse")
                      map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }else{
                      if(map_newuse_stack(list_identifier_use(i).code).head._1 != same_parent_maxid){
                         
                        builder.addEdge(list_identifier_use(i),map_newuse(list_identifier_use(i).code).head,"LastUse")
                        map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }
                    }

                     
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")

                    for(k <- 0 to list_In_Decrement.length-1){
                       
                      if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                        builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                      }
                    }
                  }
                }
              }

               

            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
 
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
 
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
          }
          else if (same_parent_maxid.code.contains("if"))
          { 
            if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
 
               
              out_control(i,this_is_out_control)
               
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
 
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
 
               
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
 
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
               
 
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
 
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
 
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
               
               
 
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
          }
          else if (same_parent_maxid.code.contains("switch"))
          {
 
            if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
               
 
              common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
               
 
              common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
            }
          }
          else if (same_parent_maxid.code.contains("="))
          {
            if (same_parent_maxid.code.contains("==")) {
               
               
 
 
 
 
 

              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
                map_newuse_stack(list_identifier_use(i).code).head._3 += list_identifier_use(next_same_name(i))
                 
                for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                  builder.addEdge(list_identifier_use(next_same_name(i)), map_newuse_stack(list_identifier_use(i).code).head._2(k) ,"LastUse")
                }
              }
            } else {
              builder.addEdge(list_identifier_use(i), list_identifier_use(next_same_name(i)), "LastUse")
              update_map_newuse(i,list_identifier_use(i))
            }
          }
          else if (same_parent_maxid.code.contains("&&"))
          {
 
             
             
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
              map_newuse_stack(list_identifier_use(i).code).head._3 += list_identifier_use(next_same_name(i))
               
              for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                builder.addEdge(list_identifier_use(next_same_name(i)), map_newuse_stack(list_identifier_use(i).code).head._2(k) ,"LastUse")
              }
            }
             
            if(map_newuse_stack.get(list_identifier_use(i).code) != None){
              var for_node = map_newuse_stack(list_identifier_use(i).code).head._1
              if(map_For_Record.get(for_node) != None){
                if(map_For_Record(for_node)(1) == true){ 
                  map_For_Second(for_node)(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                  if(!map_For_Inner(for_node).contains(list_identifier_use(i).code)){
                     
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                  }
                }
              }
            }
             
             
          }
          if(map_newuse_stack.get(list_identifier_use(i).code) != None){
 
          }else{
 
          }
          if(map_last_stack.get(list_identifier_use(i).code) != None){
 
          }else{
 
          }
        }
      }
       
    })
  }

  override def run(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    add_computedFrom(builder)
 
    add_lastuse(builder)

     
     

  }
}
