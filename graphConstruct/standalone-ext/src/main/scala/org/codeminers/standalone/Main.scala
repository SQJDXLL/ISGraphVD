package org.codeminers.standalone
import java.io._
import io.joern.x2cpg.X2Cpg.applyDefaultOverlays
//import io.shiftleft.codepropertygraph.generated.edges.ComputedFrom
import io.shiftleft.codepropertygraph.generated.{Cpg, EdgeTypes}
import io.shiftleft.codepropertygraph.generated.nodes.AstNode
import overflowdb.{Edge, Graph, Node, Property, PropertyKey}
import scala.util.control.Breaks
import java.util
import java.util.Optional
import java.nio.file.Paths

import scala.collection.mutable
//import io.shiftleft.codepropertygraph.generated.nodes.NewMynodetype
import io.shiftleft.passes.SimpleCpgPass
import io.shiftleft.semanticcpg.language._
import org.checkerframework.checker.signature.qual.Identifier
import overflowdb.BatchedUpdate
//for dot2json
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
  println("here")
  println("args", args)
  args.foreach(println)
  val variableType = args.getClass.getName
  // println("变量的数据类型是：" + variableType)
  // println(args(0),args(1))

  // 记录程序开始时间
  val startTime = System.currentTimeMillis()
  println("startTime", startTime)
  println("Hello Joern")
  print("Creating CPG... ")

  val currentPath = Paths.get(".").toAbsolutePath.normalize.toString
  println("当前执行文件的绝对路径：" + currentPath)

  val file_path: String = if (args(2).toBoolean) {  
    "../realWorld/data/" + args(0) + "/" + args(1) + "/pseudo/"  
  } else {  
    "../data/" + args(0) + "/" + args(1) + "/pseudo/"  
  }
  // val file_path =  "../data/" + args(0) + "/" + args(1) + "/pseudo/"
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
  // 打印文件数量
  println("Number of files:", file_Array.length)
  
  for(file_code <- file_Array){
    breakable {
      if (file_code.getName == ".DS_Store") {
        break()
      }

      val directory = directory_src.getAbsolutePath + '/' + file_code.getName()
      println("directory_src.getName()", directory)
      var name_of_input = file_code.getName()
      // val elements = name_of_input.split("\\.") // 注意需要转义.
      // val fileNameWithoutExtension = elements(0)
      val elements = name_of_input.split("\\.")  
      val fileNameWithoutExtension = elements.init.mkString(".")

      val config = Config(inputPaths = Set(directory))
      val cpgOrException = new C2Cpg().createCpg(config)

      cpgOrException match {
        case Success(cpg) =>
          println("[DONE]")
          println("Applying default overlays")

          applyDefaultOverlays(cpg)
          // println("Printing all methods:")
          // println("=====================")
          // println("cpg", cpg.method)
          cpg.method.name.foreach(println)
          
          println("=====================")

          var astroot: AstNode = cpg.method.l(1)

          println("Running a custom pass to add some custom nodes")
          new MyPass(cpg, astroot).createAndApply()

          val CustomGraph2Dot = new CustomGraphNodeDot(cpg.method(cpg.method.name.l(1))).dotCustomGraph.l
          // println(CustomGraph2Dot)

          val dotContent_AST: String = CustomGraph2Dot.mkString("\n")
          
          val dotContent_CFG : String = cpg.method(cpg.method.name.l(1)).dotCfg.l.mkString("\n")
          // val dotContent_CDG : String = cpg.method(cpg.method.name.l(1)).dotCdg.l.mkString("\n")

          val outputDir: String = if (args(2).toBoolean) {  
            "../realWorld/data/" + args(0) + "/" + args(1) + "/graph/" + fileNameWithoutExtension + "/"  
          } else {  
            "../data/" + args(0) + "/" + args(1) + "/graph/" + fileNameWithoutExtension + "/"  
          }
          // val outputDir = "../data/" + args(0) + "/" + args(1) + "/graph/" + fileNameWithoutExtension + "/"

          val outputFileAst = outputDir + "ast_deform.dot"
          val outputFileCfg = outputDir + "CFG.dot"
          // val outputFileCdg = outputDir + "CDG.dot"

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
          // writeToFile(outputFileCdg, dotContent_CDG)

        case Failure(exception) =>
          println("[FAILED]")
          println(exception)
      }
    }
//    // 记录程序结束时间
//    val endTime = System.currentTimeMillis()
  }

  // 记录程序结束时间
  val endTime = System.currentTimeMillis()
  println("endTime", endTime)
  // 计算程序运行时间
  val executionTime = endTime - startTime
  println(s"程序运行时间：$executionTime 毫秒")
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
//      if (n.name == "<operator>.assignment" || n.name == "<operator>.assignmentPlus" ) {
      if (list_assignment.contains(n.name)){
//        println("n",n)
        var ast_depth = n.depth
        //等号左边的identifier一定在第一层，并且每个新增的边的出点都是该点
        var computedfrom_out_node = n.astChildren.l(0)
//        println("ast",computedfrom_out_node,computedfrom_out_node.ast.l)
        var list_innodes = new ListBuffer[AstNode]()
//        if(computedfrom_out_node.isIdentifier){
//          list_innodes += computedfrom_out_node
//        }
        for(b <- computedfrom_out_node.ast.l){
          if(b.isIdentifier){
            list_innodes += b
          }
        }
//        println("list_innodes",list_innodes)
        var list_outnodes = new ListBuffer[AstNode]()
        var n_tmp = n.astChildren.l(1)
        for(a <-n_tmp.ast.l){
          if(a.isIdentifier){
            list_outnodes += a
          }
        }
//        println("list_outnodes",list_outnodes)

        for(in <- list_innodes){
          for(out <- list_outnodes){
            builder.addEdge(out, in, "ComputedFrom")
          }
        }
//        for ( i <- 1 to list_outnodes.length-1){
//          builder.addEdge(list_outnodes(0), list_outnodes(i), "ComputedFrom")
//        }


//        //获得边所有边的出点
//        var list_outnodes = ListBuffer(computedfrom_out_node)
//        var n_tmp = n.astChildren.l(1)
//        for (a <- 3 to ast_depth) {
//          if(a < ast_depth) {
//            list_outnodes += n_tmp.astChildren.l(1)
//            n_tmp = n_tmp.astChildren.l(0)
//          }
//          else{
//            list_outnodes += n_tmp.astChildren.l(0)
//            list_outnodes += n_tmp.astChildren.l(1)
//          }
//        }
//        println("list_nodes",list_outnodes)
//        print(list_outnodes.length)
        for ( i <- 1 to list_outnodes.length-1){
          builder.addEdge(list_outnodes(0), list_outnodes(i), "ComputedFrom")
//          builder.addEdge(list_outnodes(0), list_outnodes(i), "CFG")
        }
      }
    })
    //end computedfrom
  }

  def add_lastwrite(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    //    for lastwrite
    //
    //    首先查询所有的赋值结点
    cpg.call.foreach(n => {
      //保存赋值结点
      if (n.name == "<operator>.assignment") {
//        println("当前赋值语句的id",n.id)
        //首先保存lastwrite边的出结点，也就是赋值语句左边的变量的identifier结点
        var lastwrite_in = n.astChildren.l(0)
        var lastwrite_in_data = lastwrite_in.code
        //递归判断右上子树
        //只要右上跟节点存在子结点
        var n_tmp = n.ast.l(0)//没有直接n_tmp = n,因为n是call类型，将其转换为Astnode类型，方便下面的while循环更新n_tmp值
//        println("n&n_tmp",n,n_tmp)
        var flag_continue = true
        var flag_for = true
        breakable({
          //判断赋值结点的是否在循环体内（for、while）,如果源码中循环体用花括号括起来就会导致赋值结点的跟节点是block结点，再上面一层跟节点才是控制结点
          if(n.astParent.astParent.isControlStructure == true || n.astParent.isControlStructure == true){
            //判断该结点是否是循环体最右边的结点
            if(n.astParent.astParent.isControlStructure == true){
//              println("当前赋值结点在循环体内")
              //遍历循环体内所有的赋值语句组成的ast树
              var num_assigment = 0
              //记录循环体的跟节点
              var loop_rootnode = n.astParent.astParent
              var list_assignment = new ListBuffer[AstNode]()
              for(k <- n.astParent.astParent.ast.l){
                //如果为赋值语句，并且等号左右两边的变量都为n结点等号左边的变量(当前判断结点的左右子树分别包含该变量)
                if (k.code.contains("=") && k.astChildren.l(0).code.contains(lastwrite_in_data) && k.astChildren.l(1).code.contains(lastwrite_in_data)){
                  num_assigment += 1
                  list_assignment += k
                }
              }
              //n是循环体变量相关的最后一个赋值语句，包括只有一个赋值语句的情况
              //如果不是就直接按照常规的步骤进行处理
              if(n == list_assignment.last){
                //处理逻辑，从以循环体跟节点的子树，从左到右连接所有的相关变量，直到遇到第一个相关赋值语句的子树，
                // 包括该子树的相关变量在内，连接相关变量
                var flag_loop = true
                for(l <- loop_rootnode.ast.l if flag_loop){
                  //ast函数应该是按从上到下，从左到右的顺序遍历结点的，因此在遇到第一个赋值语句前，只要有同名变量就连接
                  if(l.code == lastwrite_in_data){
                    builder.addEdge(l,lastwrite_in,"LastWrite")
                  }
                  //遇到第一个赋值语句,把边加上，就不用向下寻找了
                  if(l.code.contains("=")){
                    builder.addEdge(lastwrite_in,l.astChildren.l(0),"LastWrite")
                    //找到第一个赋值语句就可以，是其本身也没关系，退出循环，停止加边行为
                    flag_loop = false
                  }
                }
              }
            }else{
//              print("粘贴一下if体内的内容")
            }
          }
          //正常情况，无论赋值语句在不在循环体里都要走该流程
          //找到当前赋值结点的下一个赋值结点，连接此期间经过的所有相关变量（变量名称相同）
          //有问题，不应该再从头开始找赋值语句，而是右上，目前代码应该是上，然后全遍历
          while(flag_continue == true) {
//            println("test_order_astnode",n_tmp.id,n_tmp)
            for (i <- n_tmp.astParent.ast.l.tail if flag_for) {
              //为了保障该赋值结点之前的同级结点不会被寻找
              if(i.id > n_tmp.id){
                //先判断是否有相同变量的identifier节点，并且需要排除n自身
                if(i.code == lastwrite_in_data && i != lastwrite_in){
//                  println("两个赋值变量之间的同名变量",i.id)
                  builder.addEdge(lastwrite_in,i,"LastWrite")
                }else if(i.code.contains("=")){
                  //找到了下一个赋值语句结点,并且赋值语句右边也有相关变量，
                  // 为了避免是x=1或者x=y这种类似情况，这几种情况立马结束当前的赋值结点n的加边，可以走向下一个赋值结点
                  for(j <- i.astChildren.l(1).ast.l){
                    if(j.code == lastwrite_in_data){
//                      println("找到的下一个相关变量存在的赋值语句",j.id)
                      builder.addEdge(lastwrite_in,j,"LastWrite")
                    }
                  }
                  break()
                }
              }
            }
            //如果向上向右一次没有找到n的下一个相关赋值语句，那么就再向上向右寻找下一个跟节点的所有ast子树
            n_tmp = n_tmp.astParent
          }
        })
      }
    })
  }

  def add_lastuse(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    //    遍历所有的identifier结点，并拿列表保存起来
    var list_identifier_use = new ListBuffer[AstNode]()
    cpg.identifier.foreach(n=>{
      list_identifier_use += n
    })

    //获取整个树的跟节点,第一个函数固定是global,因此取列表第二个元素
    breakable({

      /*
      data
       */

      //保存代码中当前每个变量的最新的数据流向,在有新边生成时，适当更新为最新的数据流向,方便了后面的节点连边
      var map_newuse = new mutable.HashMap[String,mutable.Stack[AstNode]]()

      //由于需要分析的代码结构复杂，可能存在循环体套循环体的情况，因此使用栈为每个identifier保存不同状态时的数据流
      /*
      该数据结构主要处理块嵌套快的情况，
      栈的每一层保存一个元组，
      第一个元素保存块的控制节点；
      第二个元素为列表保存入块之前的数据流（由于入块之前可能是if块，每个分支的出数据流就是入块前的数据流，可能有多个，因此使用列表保存）；
      第三个元素为列表，保存入的块的判断条件中的变量
       */
      var map_newuse_stack = new mutable.HashMap[String,mutable.Stack[(AstNode,ListBuffer[AstNode],ListBuffer[AstNode])]]()

      var map_last_stack = new mutable.HashMap[String,mutable.Stack[ListBuffer[(AstNode,AstNode)]]]()

      /*
      保存for循环内部特有的变量，因为内部特有变量与普通变量不同
       */
      var map_For_Inner = new mutable.HashMap[AstNode,ListBuffer[String]]()
      /*
      记录for第三个分支的内部变量,key为当前的for块节点，value为保存三个分支的内部变量的可变列表
       */
      var map_For_First = new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      var map_For_Second = new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      var map_For_Outer =  new mutable.HashMap[AstNode,mutable.HashMap[String,ListBuffer[AstNode]]]()
      /*
      记录当前for的三个分支分别是否存在 key: 当前for块，value：三个分支是否存在
       */
      var map_For_Record = new mutable.HashMap[AstNode,ListBuffer[Boolean]]()

      /*
      functions
      */

      //寻找给定for循环索引值对应的节点的下一个同名变量的ID,如果找不到下一个同名变量，赋值为-1
      def next_same_name(p: Int) :Int = {
        var num_true_next = p
        if(num_true_next == list_identifier_use.length-1){
          num_true_next = -1
        }else{
          num_true_next = p+1
//          println("num_true_next",num_true_next,list_identifier_use(num_true_next))
          breakable({
            while(list_identifier_use(p).code != list_identifier_use(num_true_next).code){
              num_true_next += 1
              if(num_true_next == list_identifier_use.length){//找到最后一个变量发现也没有
                num_true_next = -1
                break
              }
            }
          })
        }
        num_true_next
      }

      //寻找给定for循环索引值对应的节点的上一个同名变量的ID
      def last_same_name(p:Int) :Int = {
        var num_true_last = p-1
//        println("num_true_last", num_true_last)
        if(num_true_last <= 0){
          num_true_last = -1
        }else{
          if(list_identifier_use(num_true_last) == astroot){
            num_true_last = -1
          }
          else{
            breakable({
              while(list_identifier_use(p).code != list_identifier_use(num_true_last).code){
                if(list_identifier_use(num_true_last) == astroot){//找到跟节点也没有找到
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

      //返回两个节点的最大id公共父结点
      //思路：找到要判断的两个节点的所有父节点分别保存进两个列表，然后两层for循环直到找到两个列表中相同的节点，即为最大id公共父结点
      def same_maxid_parent(a :AstNode,b :AstNode):AstNode ={
        var list_identifier_now_parent = new ListBuffer[AstNode]()
        var list_identifier_next_parent = new ListBuffer[AstNode]()
        var now_parent = a.astParent
        var next_parent = b.astParent
//        println("cpg.method.l(1)",cpg.method.l(1))
        while(now_parent.id > cpg.method.l(1).id){
          list_identifier_now_parent += now_parent
          now_parent = now_parent.astParent
        }
        while(next_parent.id > cpg.method.l(1).id){
          list_identifier_next_parent += next_parent
//          println("next_parent",next_parent)
          next_parent = next_parent.astParent
        }
        //找到两个列表最大id的公共跟节点
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

      /*
        功能:判断identifier是否属于控制块的判断语句(不包含普通变量为入块语句的情况，该种情况有find_between_control来负责)
        思路：一直向上找父结点，直到找到是CONTROL_STRUTURE的父结点，并代表变量属于控制体或循环体（并且属于最近的CONTROL_STRUCTURE）
        若一直找到整个ast树的跟节点都没有找到说明该变量不在前面所述块里
        利用了判断语句的父节点是块节点，并且中间不会经过block.empty
       */
      def is_judge(p: Int) : (String,AstNode) = {
        var judge_for_belong:AstNode = list_identifier_use(p)
        var belong_which:String = "null"
        var belong_which_node:AstNode = judge_for_belong
        breakable{
          while(! judge_for_belong.astParent.isControlStructure){
            judge_for_belong = judge_for_belong.astParent
            //println("judge_for_belong",judge_for_belong)
            if(judge_for_belong.l.isEmpty){
              belong_which = "false"
              break
            }else if(judge_for_belong.code.contains("<empty>")){
              belong_which = "false"
//              println("judge_for_belong.astParent.code", judge_for_belong.astParent.code,judge_for_belong.astParent.isControlStructure,judge_for_belong.astParent.code.startsWith("for"))
              if(judge_for_belong.astParent.code.contains("for") && judge_for_belong.astParent.isControlStructure && judge_for_belong.astParent.code.startsWith("for")){
                if(for_Belong_Which(list_identifier_use(p),judge_for_belong.astParent) != 0){
                  //说明是for循环的前三个分支
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
            //println("why")
            belong_which = "for"
          }
          else if(belong_which_node.code.contains("switch")){
            belong_which = "switch"
          }
        }
        (belong_which,belong_which_node)
      }

      /*
        功能：判断当前变量是否为出结构体的变量
        思路：对于不同的结构体分情况讨论,应该也是从内到外找控制节点，因此列表越靠前的节点越是内层的节点
            不断通过astParent向上找控制节点，找到，若下一同名变量存在，则通过判断下一同名变量是否在以该控制节点为父节点的ast树里
       */
      def is_out_control(p:Int) :ListBuffer[(AstNode,String)] = {
//        println("当前判断的变量：",p)
        var find_parent:AstNode = list_identifier_use(p)
        var control_type:String = "true"
        var control_type_node = find_parent
        var num_loop:Int = 0
        var last_control:AstNode = control_type_node
        var control_type_map = new ListBuffer[(AstNode,String)]
        breakable{
          while(control_type != "false" && find_parent != astroot && find_parent.id > astroot.id){
            //向上找当前变量最近的一个控制结构
            if(find_parent.l(0).id > astroot.id){
              breakable({
                while(! find_parent.astParent.isControlStructure){
                  find_parent = find_parent.astParent
//                  println("find^^^^", find_parent)
                  if(find_parent.l(0) == astroot || find_parent.l(0).id < astroot.id){ //找到整个ast树的跟节点还没有找到就说明该变量不是出块变量
                    control_type = "false"
                    break
                  }
                }
              })
            }
            //判断已找到的控制变量节点为跟节点的ast树是否包含下一个同名变量的节点
            //如果本身就不存在下一个同名变量，那么该变量肯定是出结构体的变量
            if(control_type == "false" && num_loop == 0){
              //找到整个ast的跟节点也没有找到control structure
              control_type_node = find_parent
              control_type_map += ((control_type_node,control_type))
            }
            else{
              var next_node:AstNode = list_identifier_use(p)
              if(next_same_name(p) != -1){
                next_node = list_identifier_use(next_same_name(p))
              }
              find_parent = find_parent.astParent
//              println("find_control-structure",find_parent)
              control_type_node = find_parent
              if(find_parent.code.contains("if") || find_parent.code.contains("else")){
                if(find_parent.code.contains("if")){ //对于if的情况
//                  println("if-order_or_nest",find_parent,last_control,order_or_nest_plus(find_parent,last_control))
                  if(num_loop > 0 && order_or_nest_plus(find_parent,last_control) == 5)
                  {
                    //找到的控制节点和与当前要判断的变量所在的分支是顺序关系
//                    println("if-here")
                    break() //顺序分支关系
                    //这个break没有成功推出循环，因此出块变量判断出错
                    //println("if-here=here")
                  }
                  else
                  {
                    if(next_same_name(p) == -1){ //当前变量已经是函数中最后一个同名变量了
                      control_type = "if"
                      last_control = control_type_node//保存上一轮保存的出块节点
                      control_type_map += ((control_type_node,control_type))
                      num_loop +=1
                    }
                    else if(find_parent.astChildren.l(0).ast.l.contains(next_node) || find_parent.astChildren.l(1).ast.l.contains(next_node)){
                      control_type = "false"
                    }
                    else{
                      control_type = "if"
                      last_control = control_type_node//保存上一轮保存的出块节点
                      control_type_map += ((control_type_node,control_type))
//                      if(find_parent.astParent.astParent.code.contains("if")){ //为了当前出块是if分支的第一个分支考虑
//                        find_parent = find_parent.astParent.astParent
//                      }
//                      else if(find_parent.astParent.astParent.code.contains("else")){
//                        find_parent = find_parent.astParent.astParent.astParent
//                      }
                      if(find_parent.astParent.astParent.code.contains("else")){
                        if(find_parent.astParent.astChildren.l.length == 1) {
                          //只能暂时先这么判断了，因为else if和else{if()}目前的结构都是else-block-if，智能根据block的一阶子节点的个数来判断
                          find_parent = find_parent.astParent.astParent.astParent
                        }
                      }
//                      println("find_parent",find_parent)
                      num_loop +=1
                    }
                  }
                }
                else
                {//对于else的情况
                  //else分支下的control_structure肯定与其是嵌套关系，直接用is_father来判断
                  //println("else-order_or_nest",find_parent,last_control,order_or_nest(find_parent,last_control))
                  //                  if(num_loop > 0 && order_or_nest(find_parent,last_control)){
                  if(num_loop > 0 && !is_father(find_parent,last_control)){
                    //println("here out")
                    break() //顺序分支关系
                  }
                  else{
                    if(next_same_name(p) == -1){
                      control_type = "if"
                      last_control = control_type_node//保存上一轮保存的出块节点
                      control_type_map += ((control_type_node,control_type))
                      num_loop += 1
                    }
                    //在else的时候只有一个ast分支
                    else if(find_parent.astChildren.l(0).ast.l.contains(next_node)){
                      control_type = "false"
                    }
                    else{
//                      println("in here",control_type_node)
                      control_type = "if"
                      last_control = control_type_node//保存上一轮保存的出块节点
                      control_type_map += ((control_type_node,control_type))
                      //                      if(find_parent.astParent.code.contains("if")){
                      //                        println("000000000")
                      //                        find_parent = find_parent.astParent
                      //                        println("00000",find_parent)
                      //                      }
                      num_loop +=1
                    }
                  }
                }
              }
              else if(find_parent.code.contains("for") && find_parent.code.startsWith("for")){
                //其他结构不需要判断是否与上一轮找到的块是否是分支还是嵌套关系，因为不可能隔着一个多分支相关的块是其块的另一个分支的嵌套的块
                if(next_same_name(p) == -1){
                  control_type = "for"
                  last_control = control_type_node//保存上一轮保存的出块节点
                  control_type_map += ((control_type_node,control_type))
                  num_loop +=1
                }
                else if(find_parent.ast.l.contains(next_node)){
                  control_type = "false"
                }else{
                  control_type = "for"
                  last_control = control_type_node//保存上一轮保存的出块节点
                  control_type_map += ((control_type_node,control_type))
                  num_loop +=1
                }
              }
              else if(find_parent.code.contains("while")){
                if(find_parent.code.contains("do")){ //do-while
                  if(next_same_name(p) == -1){
                    control_type = "do-while"
                    last_control = control_type_node//保存上一轮保存的出块节点
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                  else if(find_parent.astChildren.l(0).ast.l.contains(next_node) || find_parent.astChildren.l(1).ast.l.contains(next_node)){ //block块里没有
                    control_type = "false"
                  }else{
                    control_type = "do-while"
                    last_control = control_type_node//保存上一轮保存的出块节点
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                }else{ //while
                  if(next_same_name(p) == -1){
                    control_type = "while"
                    last_control = control_type_node//保存上一轮保存的出块节点
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                  else if(find_parent.ast.l.contains(next_node)){
                    control_type = "false"
                  }else{
                    control_type = "while"
                    last_control = control_type_node//保存上一轮保存的出块节点
                    control_type_map += ((control_type_node,control_type))
                    num_loop +=1
                  }
                }
              }
              else if(find_parent.code.contains("switch")){
                //println("find_switch")
                if(next_same_name(p) == -1){
                  control_type = "switch"
                  //                  last_control = control_type_node//保存上一轮保存的出块节点
                  last_control = find_belong_case(p,control_type_node)
                  control_type_map += ((last_control,control_type))
                  num_loop +=1
                }
                else{
                  //                  println("herehete")
                  //还需要排除当前变量是switch的判断语句，即switch左边的子树包含当前变量
                  if(! find_parent.astChildren.l(0).ast.l.contains(list_identifier_use(p))){
                    if(find_parent.astChildren.l(1).ast.l.contains(next_node)){
                      //                    println("here222")

                      val list_range = new ListBuffer[AstNode]()
                      //如果两个同名变量中间夹着case或default，代表是出了一个分支
                      breakable{
                        for (i<-find_parent.astChildren.l(1).ast.l){ //第二个分支才是正式进入switch的块
                          if(i.id > list_identifier_use(p).id && i.id < next_node.id){//通过当前变量的id和下一同名变量的id限制寻找的范围
                            //println("i.code",i.code)
                            //如果下一同名变量在子树里，按照两个变量之间是否有case或default来判断
                            if(i.code.contains("case") || i.code.contains("default")){
                              if(i.astParent.astParent == find_parent){
                                //  并且判断找到的case或default是否是当前switch的二阶子节点，因为还存在switch嵌套switch的情况
                                //println("find_case")
                                control_type = "switch"
                                //注释的方法保存的是大的switch节点，而不是case或default节点
                                //                                last_control = control_type_node//保存上一轮保存的出块节点
                                //                                control_type_map += ((control_type_node,control_type))
                                last_control = find_belong_case(p,i.astParent.astParent)//保存上一轮保存的出块节点
//                                println("last_congtrol",last_control)
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
                      last_control = control_type_node//保存上一轮保存的出块节点
                      control_type_map += ((control_type_node,control_type))
                      num_loop += 1
                    }
                  }
                }
                if(control_type != "switch"){//如果全部遍历了也没有case或default就说明不是出结构
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

      /*
      该函数只针对switch使用
      给出一个节点和他所在的switch节点，返回该元素所属的switch的case或default分支
      思路：可以直接使用find_some_node函数，找出当前变量和其上一同名变量之间的case或default，返回的是离当前变量最近的，也就是其所属的分支
          但是存在上一同名变量不存在的情况，
       */
      def find_belong_case(p:Int,case_node:AstNode) : AstNode = {
        var node_p : AstNode = list_identifier_use(p)
        var find_node :AstNode = node_p

        //使用switch节点和当前节点中寻找
        if(find_some_node(case_node,node_p,"case")._1){
          find_node = find_some_node(case_node,node_p,"case")._2.last
        }
        if(find_some_node(case_node,node_p,"default")._1){
          find_node = find_some_node(case_node,node_p,"case")._2.last
        }

        find_node
      }

      /*
      功能：判断一个变量的下一个同名变量是否在赋值语句的左侧，且赋值号右侧有同名变量
      思路：如果当前变量x的下一同名变量y的同名变量z,若y与z的公共父结点是<operator>.assignment
       */
      def assignment_right_have(p:Int):Boolean = {
        var is_assignment_right_have: Boolean = false
        if(next_same_name(p) != -1 && next_same_name(next_same_name(p)) != -1){
          if(same_maxid_parent(list_identifier_use(next_same_name(p)),list_identifier_use(next_same_name(next_same_name(p)))).code.contains("=")){
            is_assignment_right_have = true
          }
        }
        is_assignment_right_have
      }

      /*
      功能：当赋值号两边都只有一个变量，判断该变量是左边还是右边
      思路：根据节点的id的大小
      输入：当前节点的
      返回：true代表在左侧,false代表在右侧
       */
      def left_right(p:Int):Boolean = {
        var id_left: Long = list_identifier_use(p).astParent.astChildren.l(0).id
        var is_left :Boolean = true
        if(list_identifier_use(p).id != id_left){
          is_left = false
        }
        is_left
      }

      /*
      功能：判断是否更新了数据流（x=1,代表更新了数据流；x=x+y,y=x，func(x,y)函数调用操作不算更新数据流）
      思路：从当前变量向上找（astParent），直到找到第一个assignment,
            if(找到assignment)然后看该节点的code包不包含这个节点，
              并根据当前节点的id与找到的assignment节点的一阶右侧子节点的id大小判读变量是在赋值语句的左侧还是右侧（如果大于等于说明在右侧，否则在左侧）
              如果在左侧，并且与下一同名变量的公共父节点是assignment,说明是左右都有，如果公共父节点不是说明被重新赋值（即数据流更新了）
              如果在右侧就是默认数据流没有被更新
            else(找到astroot都没有找到assignment)默认数据流没有被更新（一些函数等操作 ，例如调用函数的参数也是identifier）
       */
      def update_data(p:Int): (Boolean,Int) = {
        var update_data:Boolean = true
        var find_assignment:Boolean = false
        // var p_node : AstNode = list_identifier_use(p)
        // if p_node != astroot && p_node.astParent
        var p_node: AstNode = list_identifier_use(next_same_name(p)).astParent
        var false_type:Int = 0 //为了区别是那种情况的未更新数据流
        var p_id:Long = list_identifier_use(next_same_name(p)).id
        breakable{//找到
          while(p_node != astroot && p_node.id > astroot.id){
            //由于do-while结构会把块中所有的代码放在do-while节点上，很容易有等于号等符号，所以还要再判断不是控制节点
            if(((p_node.code.contains("=") && !p_node.code.contains("=="))|| p_node.code.contains("++") || p_node.code.contains("--")) && ! p_node.isControlStructure){
              find_assignment = true
              break
            }
            // println("p_node", p_node, p_node.code, p_node.id)
            p_node = p_node.astParent
          }
        }

        if(find_assignment){
          //如果找到了
          if(p_id >= p_node.astChildren.l.last.id){
            //println("is_here",p_node.astChildren.l.last)
            update_data = false //y=x
            false_type = 1
            var List_assignment_Self = ListBuffer[String]("++","--")
            breakable{
              for(type_string <- 0 to List_assignment_Self.length-1){
                //println("p_node",p_node.code)
                if(p_node.code.contains(List_assignment_Self(type_string))){//无论++在变量的左侧还是右侧
                  update_data = false //x++ ++x x-- --x
                  false_type = 3
                  break
                }
              }
            }
          }else{
            if(next_same_name(p) != -1 && next_same_name(next_same_name(p)) != -1){
              //println("same_maxid_parent(list_identifier_use(p),list_identifier_use(next_same_name((p))))",same_maxid_parent(list_identifier_use(p),list_identifier_use(next_same_name((p)))))
              if(same_maxid_parent(list_identifier_use(next_same_name(p)),list_identifier_use(next_same_name((next_same_name(p))))) == p_node){
                update_data = false //x=x+y
                false_type = 2
              }
              else{
                update_data = true //x=1
                var List_assignment = ListBuffer[String]("+=","-=","*=","/=")
                breakable{
                  for(type_string <- 0 to List_assignment.length-1){
                    if(p_node.code.contains(List_assignment(type_string)) && p_id < p_node.astChildren.l.last.id){//变量需要在自增符号左侧
                      update_data = false //x+=a
                      false_type = 3
                      break
                    }
                  }
                }
              }
            }
            else{
              update_data = true //x=1
              var List_assignment = ListBuffer[String]("+=","-=","*=","/=")
              breakable{
                for(type_string <- 0 to List_assignment.length-1){
                  if(p_node.code.contains(List_assignment(type_string)) && p_id < p_node.astChildren.l.last.id){//变量需要在自增符号左侧
                    update_data = false //x+=a
                    false_type = 3
                    break
                  }
                }
              }
            }
          }
        }else{
          //如果没有找到
          update_data = false //func(x,y)
          false_type = 4
        }
        //println("update_data",p_id,update_data,false_type)
        (update_data,false_type)
      }

      /*
      功能：判断一个节点是否是另一个节点的父结点，可以多跳
      参数：已经在栈里的节点(id小)，要新入栈的节点（id大）
      返回值为true说明p是q的父节点（多级）
       */
      def is_father(p:AstNode, q:AstNode):Boolean = {
        var q_tmp:AstNode = q
        var is_father:Boolean = false
//        println("id",p.id,q.id)
        if(p.id < q.id){ //p若是q的父节点，p的id一定小于q的id,若不是则直接返回false
          breakable{
            while(p != q_tmp){
//              println("q_tmp",q_tmp)
              q_tmp = q_tmp.astParent
              //println("q_tmp",q_tmp)
              if(q_tmp.id < p.id){
                break
              }
            }
            is_father = true
          }
        }
        is_father
      }

      /*
      执行更新map_newuse栈保存数据流的操作
       */
      def update_map_newuse(i:Int,node:AstNode) :Unit ={
        if (map_newuse.get(list_identifier_use(i).code) == None) {
          map_newuse(list_identifier_use(i).code) = Stack(node)
        } else {
          map_newuse(list_identifier_use(i).code).pop()
          map_newuse(list_identifier_use(i).code).push(node)
        }
      }

      /*
        执行连边操作
       */
      def draw_line(i:Int,node:AstNode) :Unit ={
        if (map_newuse.get(list_identifier_use(i).code) == None || map_newuse(list_identifier_use(i).code).isEmpty) {
          builder.addEdge(node, list_identifier_use(i), "LastUse")
        } else {
          builder.addEdge(node, map_newuse(list_identifier_use(i).code).head ,"LastUse")
        }
      }

      /*
        判断当前结构体是否处于其他结构体内（即是否被嵌套）
       */
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

      /*
      功能；给定节点id，向上寻找当前节点的最近的control节点
       */
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

      /*
      判断两个if分支的结构体节点是属于同一if块，还是嵌套的关系（可以跨级）
      思路：看出分支或出块的control节点是否为当前map_last_stack栈顶的列表的第一个控制节点的之后的属于同一if的控制节点
      p:map_last_stack的节点，q为出分支或出块的节点(p为id小的节点，q为id大的节点)
      order_or_nest:true代表顺序分支关系，false代表嵌套关系
       */
      //只需要评判是否为嵌套关系
      //处理的仅是if和if分支的关系，是嵌套，还是顺序的关系

      def order_or_nest(p:AstNode,q:AstNode) : Boolean = {
        var order_or_nest : Boolean = false
        var p_children_option:Option[AstNode] = p.astChildren.l.lastOption
        if(!p_children_option.isEmpty){
          var p_children:AstNode = p.astChildren.l.last
          //碰到else分支，或没有子节点了
          breakable{
            while(p_children.astChildren.l.lastOption != None){
              if(q.id < p_children.id){ //在前两个分支，所以属于嵌套关系
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

      /*
      判断任意两个if多分支的分支节点的关系
       */
      def order_or_nest_if(p:AstNode,q:AstNode) : Int = {
        //默认p的id小于q的id，使用函数之前可以通过条件判断来调换参数的位置
        /*
        1 嵌套 （q在的整个多分支嵌套于p中）
        2 串行  （p和q没有关系，就是普通的串行关系,即q没有在p内）
        3 上行分支嵌套,实际无 (不存在该种情况，因为p的id小于q)
        4 下行分支嵌套 （q在p的下行分支内部，属于嵌套）
        5 同一if块的不同分支（p是q的上行分支，即他们属于同一个if多分支内）
         */

        var relation :Int = 0
        if(is_father(p,q)){
          var p_children_option:Option[AstNode] = p.astChildren.l.lastOption
          var num_loop : Int = 0
          if(!p_children_option.isEmpty){
            var p_children:AstNode = p.astChildren.l.last
            //碰到else分支，或没有子节点了
            breakable{
              while(p_children.astChildren.l.lastOption != None){
                if(q.id < p_children.id){ //在前两个分支，所以属于嵌套关系
                  if(num_loop == 0){
                    //如果第一次循环，q就在p的前两个分支里
                    relation = 1
                    break
                  }else{
                    //说明，q在p的下行分支中
                    relation = 4
                    break
                  }
                }
                else if(p_children == q){
                  relation = 5
                  break
                }else{
                  //q.id > p_children.id的情况属于p是else分支，q在p的子节点的内部，else只有一个分支,所以也是嵌套关系
                  relation = 1
                }
                num_loop += 1
                p_children = p_children.astChildren.l.last
              }
            }
          }
        }else{
          //q不在p中，所以他们就是普通的串行关系
          relation = 5
        }
        relation
      }

      //给出switch分支的关系，
      // 包括 同一switch的不同分支(同级)、不同switch之间（不同switch之间包括： switch嵌套switch（即一个switch在另一个switch的某个分支里），和 switch之间是串行的关系）

      /*
      1 嵌套  （q所在的switch在p这个case分支中）
      2 串行  完全不相关
      3 上行分支嵌套（不存在）
      4 下行分支嵌套  在与p同级的下行的其他case中
      5 同一switch的不同分支
       */
      //p的id小，q的id大，p，q均为case或default节点
      def order_or_nest_switch(p:AstNode,q:AstNode) : Int = {
        var relation : Int = 0
        var p_q_parent : AstNode = same_maxid_parent(p,q)
        var p_switch : AstNode = p.astParent.astParent
        var q_switch : AstNode = q.astParent.astParent
        if(p_switch == q_switch){
          //1
          relation = 5 //同一switch的case分支
        }
        else{
          var is_find : Boolean = false
          if(is_father(p_switch,q_switch)){
            //2
            //找到两个节点之间所有的case和default,并且筛选出与p这个case同级的case或default节点，如果筛选出来了节点，说明q所在的case的switch不在p代表的分支里
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
              relation = 4 //在与p同级的别的case中
            }
            else{
              relation = 1 //在p分支中
            }
          }else{
            //3
            relation = 2 //完全不相关
          }
        }
        relation
      }


      //判断的是多分支块与单分支块的关系
      /*
      1 多分支与单分支的关系
        多分支的某个分支节点与单分支的关系：
          1.1 多分支的单个分支是单分支的父节点
           1.1.1 多分支某一分支嵌套单分支
           1.1.2 多分支的某一分支的下行顺序分支嵌套单分支（不可能为上行顺序分支）

           判断该多分支的某一分支的所有下行顺序分支为父节点的第二分支子节点（需要具体考虑最后一个else的情况，也可能没有else）是否包含该单分支，
           若某一下行分支包含，属于1.1.2,没有任一下行分支包含，则属于1.1.1

          1.2 单分支是多分支父节点 （单分支包含整个多分支）
          1.3 除了前两种情况（他们没有关系，在代码里表现就是串行）
      2 多分支与多分支的关系
        order_or_nest解决了问题
        但是还有switch的情况
        //多分枝的某一分支嵌套多分支
      3 单分支与单分支的关系  嵌套与非嵌套关系
        若一方是另一方的父节点，那么为嵌套关系，否则他们没有关系（在代码里表现就是串行）
       */

      /*
      默认p的id比q的id小
      p和q均为代表了块或者分支的control structure节点，switch节点除外（switch块目前使用case和default）
      所以他们的关系分类主要有 1 嵌套 2 无关(串行) 3 该多分支的某一分支的上行顺序分支嵌套多分枝或者单分支 4 该多分支的某一分支的下行顺序分支嵌套整个多分支或者单分支 5 只存在于多分支的情况，即他们是属于同一多分支块的不同分支
       */
      def order_or_nest_plus(p:AstNode,q:AstNode) : Int = {
        var relation : Int = 0
        var p_branch : Boolean = true //true代表单分支
        var q_branch : Boolean = true //true代表单分支


        //根据节点的code来判断是单分支还是双分支,true表示单分支
        def multi_Or_Single(node:AstNode):Boolean = {
          var multi_or_single: Boolean = true
          if((node.code.contains("for") && node.code.startsWith("for")) || node.code.contains("while")){ //for do-while while
            multi_or_single = true
          }
          else if(node.code.contains("if") || node.code.contains("else") || node.code.contains("switch") || node.code.contains("case") || node.code.contains("default")){
            multi_or_single = false
          }
          multi_or_single
        }

        p_branch = multi_Or_Single(p)
        q_branch = multi_Or_Single(q)

        //println("p_branch,q_branch",p_branch,q_branch)

        def mutliFirst() : Unit = {
          var p_q_parent : AstNode = same_maxid_parent(p,q)
          //if和switch分情况讨论
          //switch
          if(p.code.contains("case") || p.code.contains("default")){
            if(p_q_parent.astParent.isControlStructure && p_q_parent.astParent.code.contains("switch")){
              //判断是否是嵌套还是上行的关系
//              println("inninin")
              var is_find : Boolean = false
              for(node <- find_some_node(p,q,"case")._2){
                if(node.astParent.astParent == p.astParent.astParent){
//                  println("node",node)
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
                relation = 1//之前为什么是3
              }
            }else{
              relation = 2 //多分支的分支与单分支不在一个switch里
            }
          }else if(p.code.contains("if") || p.code.contains("else")){
            //if
            if(is_father(p,q)){
              //println("herehere")
              //println("p.code",p.code)
              if(p.code.contains("if")){
                if(p.astChildren.l(1).ast.l.contains(q)){
                  relation = 1
                }else{
                  //下行
                  relation = 4
                }
              }else if(p.code.contains("else")){
                //println("11111111")
                relation = 1 //一定是嵌套
              }
            }else{
              relation = 2
            }
          }
        }
        if(p_branch && q_branch){ //单单
          if(is_father(p,q)){
            relation = 1
          }else{
            relation = 2
          }
        }
        else if(p_branch && !q_branch){
          //单多，单分支的id小，多分支的id大
          //单分支的id小，多分支的id大，那么有两种情况
          // 1 单分支嵌套整个多分支，这种情况下，单分支节点是多有多分支的分支节点的多级父节点
          // 2 单分支与多分枝没有关系,也就是串行的关系
          // 3 单分支在该多分支的上行顺序分支中 if(){while}else if(){},即while和else if的关系
          if(is_father(p,q)){
            relation = 1
          }else{
            relation = 2
            if(q.code.contains("if") || q.code.contains("if")){
              if(same_maxid_parent(p,q).isControlStructure && same_maxid_parent(p,q).code.contains("if")){
                //如果p和q的公共父节点是if,说明是上行分支
                relation = 3
              }
            }else if(q.code.contains("case") || q.code.contains("default")){
              //如果在两个节点的公共父节点是block，且block的一阶父节点是switch,那么说明p在q的上行分支
              var p_q_parent: AstNode = same_maxid_parent(p,q)
              if(p_q_parent.isBlock && p_q_parent.astParent.code.contains("switch")){
                relation = 3
              }
            }
          }
        }
        else if(!p_branch && q_branch){ //多单 多分支的id小，单分支的id大
          /*
           多分支的id小，单分支的大，应该有三种情况
           1 多分支的分支与单分支没有关系，即串行关系
              根据公共父节点判断
           2 单分支在多分支分支节点的分支内，但不一定在当前多分支节点的分支内，可能有两种关系，一种是嵌套，一种是在下行分支中
              2.1 嵌套
              2.2 下行分支
           */
          mutliFirst()
        }
        else if(!p_branch && !q_branch){ //多多
          //println("p.code,q.cpde",p.code,q.code)
          if((p.code.contains("if") || p.code.contains("else")) && (q.code.contains("if") || q.code.contains("else"))){
            relation = order_or_nest_if(p,q)
          }else if((p.code.contains("case") || p.code.contains("default")) && (q.code.contains("case") || q.code.contains("default"))){
            relation = order_or_nest_switch(p,q)
          }else if((p.code.contains("if") || p.code.contains("else")) && (q.code.contains("case") || q.code.contains("default"))){
            mutliFirst()
          }else if( (p.code.contains("case") || p.code.contains("default")) && (q.code.contains("if") || q.code.contains("else"))){
            //println("case if")
            mutliFirst()
          }
        }
        relation
      }

      /*
      判断当前变量在for块的哪个分支
      p代表当前被判断的变量，q代表for块的control structure节点
       */
      def for_Belong_Which(p:AstNode,q:AstNode) : Int = {
        var for_Belong_Which : Int = 0
        //并把空格去掉
        var for_string :String = q.code.split("for").last.replace(" ", "") //for(;;)
        var for_string_list = for_string.split(";")
        var for_string_hashmap = new mutable.HashMap[Int,Boolean]() //记录三个分支是否为空
        var num_true = 0
        var for_real = new mutable.HashMap[Int,Int]() //记录真实的分支顺序，key为for中的分支顺序，value为树中真实的分支顺序
        for(k <- 0 to for_string_list.length-1){
          if(for_string_list(k).length > 1){
            for_string_hashmap(k+1) = true //不为空，即当前位置有元素
            num_true += 1
          }else{
            for_string_hashmap(k+1) = false
          }
        }
        //println("for_string_hashmap",for_string_hashmap)
        var num_real = 0
        for( k <- 0 to for_string_list.length-1 if(for_string_hashmap(k+1)==true)){
          num_real += 1
          for_real(k+1) = num_real //value为树中真正的顺序
        }
        //println("for_real",for_real)
        breakable{
          for(m <- 0 to 2){
            if(for_string_hashmap(m+1) == true){
              //if()
              if(q.astChildren.l(for_real(m+1)-1).ast.l.contains(p)){
                for_Belong_Which = m+1
                break
              }
            }
          }
        }
        //println("for_belong_which",for_Belong_Which)
        for_Belong_Which
      }

      /*
      为嵌套关系的if找到其外层if节点
      向上找父节点直到找到代码包含if或else的节点
       */
      def find_outer_if(p:AstNode):AstNode = {
        var p_parent:AstNode = p.astParent
        while((p.astParent.code.contains("if"))||(p.astParent.code.contains("else"))){
          //向上找先碰到if后遇到else
          p_parent = p_parent.astParent
        }
        p_parent
      }

    /*
      判断两个给定节点中间是否包含某种类型的节点
      该函数一般用于寻找两个节点之间是否包含break、continue、以及switch结构中的case和default等关键节点
      返回一个元组，元组的第一个元素为是否包含指定类型的节点，第二个元素为若包含，则返回找到的节点，若不包含，则返回参数传入的第一个节点

      由于是从下一同名变量的父节点开始从右向左，从下到上的寻找，并且找到要找的元素的第一个节点就停止,所以要找的节点如果有多个，返回的是离下一同名变量最近的节点
     */
      def find_some_node(p:AstNode,q:AstNode,find_string:String):(Boolean,ListBuffer[AstNode]) = {
        /*
        从节点大的q点开始找，
         */
        //println("p,q",p,q)
        var is_find :Boolean = false
        var find_node = new ListBuffer[AstNode]
        var q_father: AstNode = q.astParent
        //var num_iter : Int = 0
        var q_father_Previous_Round : AstNode = q
        //需要减去重复的ast

        while(q_father != same_maxid_parent(p,q).astParent){
          for(node <- q_father.ast.l if (node.id < q_father_Previous_Round.id && node.id > p.id)){
            //println("node_out",node)
            if(node.code.contains(find_string)){
              if(node.isControlStructure){
                //因为do-while这种结构会把代码中的内容放到节点的code中
                if(find_string == "break" || find_string == "continue"){
                  is_find = true
                  find_node += node
                  //break
                }
              }
              else{//针对switch的case和default情况，这两种节点都不是control_structure
                is_find = true
                //println("node",node)
                find_node += node
                //println("find_node",find_node)
                //break
              }
            }
          }
          q_father_Previous_Round = q_father
          q_father = q_father.astParent
          //          if(!is_find){
          //            q_father = q_father.astParent
          //          }
          //          else{
          //            break
          //          }
        }

//        while(q_father != same_maxid_parent(p,q).astParent){
//          for(node <- q_father.ast.l if (node.id < q.id && node.id > p.id)){
//            println("node_out",node)
//            if(node.code.contains(find_string)){
//              if(node.isControlStructure){
//                //因为do-while这种结构会把代码中的内容放到节点的code中
//                if(find_string == "break" || find_string == "continue"){
//                  is_find = true
//                  find_node += node
//                  //break
//                }
//              }
//              else{//针对switch的case和default情况，这两种节点都不是control_structure
//                is_find = true
//                println("node",node)
//                find_node += node
//                //println("find_node",find_node)
//                //break
//              }
//            }
//          }
//          q_father_Previous_Round = q_father
//          q_father = q_father.astParent
////          if(!is_find){
////            q_father = q_father.astParent
////          }
////          else{
////            break
////          }
//        }
        //println("is_find,find_node",is_find,find_node)
        (is_find,find_node)
      }

      /*
      Break和continue都属于control structure

      其实break和continue的存在，就是将其同样视为一个出结构体情况就可以了，
      但是他们两个还是有区别的：
      对于break来说，其作为出块情况，只会影响出块的下一个同名变量，也就是下一同名变量也要考虑该出块数据流
      对于continue来说，作为出块的情况来说，不仅会影响下一个同名变量，还会影响当前块的判断语句的同名变量。

      两个变量之间若是有break,将当前变量作为出数据流入map_last_stack栈，出的块节点通过is_out_control找到
       */


      /*
      目的：为了找出不在判断语句中的变量充当入块角色时，可能入的一些块（着重判断的是下一同名变量是都是普通语句的情况下也是入块的变量）
      功能：找到两个节点（当前变量与下一同名变量）之间的相关块节点，这里"相关"的定义指为下一同名变量为普通变量时寻找可能为某些块的入块变量的情况
      思路：从下一同名变量的父节点开始找，找到控制节点，直到找到两个节点的公共父节点
      参数：i为当前变量的序号
       */
//      def find_between_control(i:Int):ListBuffer[AstNode] = {
//        var find_result = new ListBuffer[AstNode]
//        var q_parent:AstNode = list_identifier_use(next_same_name(i)).astParent
//        //直到找到公共父节点，结束中间所有块的寻找
////        println("q_parent",q_parent)
//        println("same_parent",same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))))
//
//        while(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
////          println("q_parent",q_parent)
//          if(q_parent.isControlStructure)
//          {
//            if(q_parent.code.contains("while")){
//              if(q_parent.code.contains("do")){//do while结构 的control_structure节点的内容也是
//                find_result += q_parent
//                q_parent = q_parent.astParent
//              }else{//while结构
//                //直接记入find_result
//                find_result += q_parent
//                q_parent = q_parent.astParent
//              }
//            }
//            else if(q_parent.code.contains("for") && q_parent.code.startsWith("for")){
//              find_result += q_parent
//              q_parent = q_parent.astParent
//            }
//            else if(q_parent.code.contains("if")){
////              println("ifififiif")
//              find_result += q_parent
//              breakable{
//                while(q_parent.code.contains("if")){
//                //该判断有两种可能的情况，一种是else if（也是为了排除这种情况），另一种是else{if()},但是两种情况在树上面表现一致
//                  //目前只能根据第一种情况block的子节点只有if，第二种情况block节点可能有多个（但也可能只有一个，使用这种方法就会失效，目前没有找到区别两种情况更好的办法）
//                  //暂时用block的一阶子节点的个数区分
//                  if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else") && q_parent.astParent.astChildren.l.length == 1){
////                    println("^^^^^^^^^^^^^^^")
//                    q_parent = q_parent.astParent.astParent.astParent //跳过该不需要的控制节点
//                    break()
//                  }else{
////                    println("*****************")
//                    q_parent = q_parent.astParent
//                    break()
//                  }
////                  if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else")){
////                    q_parent = q_parent.astParent.astParent //跳过该不需要的控制节点
////                  }else{
////                    q_parent = q_parent.astParent
////                    break()
////                  }
//                }
//              }
//            }
//            else if(q_parent.code.contains("else")){
////              println("elseelseelse")
//              find_result += q_parent
//              q_parent = q_parent.astParent //根据结构找到else的直接父节点if
////              println("q_parent_jump _if",q_parent)
//              if(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
//                breakable{
//                  while(q_parent.code.contains("if")){
//                    if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else")){
//                      //并且else节点的子节点的个数只有一个（目前只能这样子）
////                      println("herehere",q_parent.astParent.astParent)
////                      println(q_parent.astParent.astParent.astChildren.l)
//                      if(q_parent.astParent.astParent.astChildren.astChildren.l.length == 1){
////                        println("1111111")
//                        q_parent = q_parent.astParent.astParent.astParent //跳过该不需要的控制节点
//                        break()
//                      }else{
////                        println("lenth>1")
//                        q_parent = q_parent.astParent
////                        println("q_parent",q_parent)
//                        break()
//                      }
//                    }else{
//                      q_parent = q_parent.astParent
//                      break()
//                    }
//                  }
//                }
//              }
//            }
//            else if(q_parent.code.contains("switch")){//针对switch的情况
//              //switch反到不用像if那样复杂，因为switch的每个分支的二级父节点直接是switch这个control节点
//              /*这种情况是
//                2 switch外到switch内的判断语句这种情况要排除（？？？）
//                3 switch外到switch内的分支语句
//               */
//              //println("find_switch")
//              if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
////                println("find_case111")
////                println(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2)
//                find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
//              }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
////                println("find_default")
//                find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
//              }else{//普通变量入switch的判断语句情况
////                println("putomngruswitch")
//                find_result += q_parent
//              }
////              find_result += q_parent
//              q_parent = q_parent.astParent
//            }
//          }
//          else{
////            println("q_parnet_front",q_parent)
////            println(q_parent.astParent.id, cpg.method.l(1).id)
////            if(q_parent.astParent.id > cpg.method.l(1).id) {
//            q_parent = q_parent.astParent
////            }
////            println("查看一下应该等于啥",q_parent)
//          }
////          println("q_parent_last",q_parent)
//          //println("q_parent",same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))),q_parent)
//        }
//
//        //应对的是switch的情况
//        if(q_parent == same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
//          //println("am i here")
//          if(q_parent.astParent.code.contains("switch")){
//            //println("am i here1",q_parent.astParent.code,same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))))
//            //应对的是同一个switch的不同分支的情况
//            //思路：如果两个变量之间有case或default说明下一同名变量是swicth某一分支的入变量
//            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
//              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
//            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
//              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
//            }
//
//          }else if(q_parent.isControlStructure && q_parent.code.contains("switch")){
//            //println("am i here2")
//            //应对的是switch的判断语句和不同分支
//            //思路，也是寻找离下一同名变量节点最近的case和default
//            //println("16,21")
//            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
//              //println("16,21",find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case"))
//              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
//            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
//              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
//            }
//          }
//        }
//
//
//        //下面的思路不太对吧
////        if(q_parent == same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))) && q_parent.isControlStructure){
////          //因为前有一个退出判断的条件是当判断到两个变量的公共父节点时，像while和for都有可能是中间没有同名变量，寻找中间的控制节点只有自己
////          find_result += q_parent
////        }
//        //println("find_between_control",find_result)
//        find_result
//      }

      def find_between_control(i:Int):ListBuffer[AstNode] = {
        var find_result = new ListBuffer[AstNode]
        var q_parent:AstNode = list_identifier_use(next_same_name(i)).astParent
        //直到找到公共父节点，结束中间所有块的寻找
        //        println("q_parent",q_parent)
//        println("same_parent",same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))))
//        print("q_parent", q_parent)
        breakable{
          while(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
            //          println("q_parent",q_parent)
            if(q_parent.isControlStructure)
            {
              if(q_parent.code.contains("while")){
                if(q_parent.code.contains("do")){//do while结构 的control_structure节点的内容也是
                  find_result += q_parent
                  if(q_parent.astParent.id > cpg.method.l(1).id){
                    q_parent = q_parent.astParent
                  }else{
                    break
                  }
                }else{//while结构
                  //直接记入find_result
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
                //              println("ifififiif")
                find_result += q_parent
                breakable{
                  while(q_parent.code.contains("if")){
                    //该判断有两种可能的情况，一种是else if（也是为了排除这种情况），另一种是else{if()},但是两种情况在树上面表现一致
                    //目前只能根据第一种情况block的子节点只有if，第二种情况block节点可能有多个（但也可能只有一个，使用这种方法就会失效，目前没有找到区别两种情况更好的办法）
                    //暂时用block的一阶子节点的个数区分
                    if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else") && q_parent.astParent.astChildren.l.length == 1){
                      //                    println("^^^^^^^^^^^^^^^")
                      q_parent = q_parent.astParent.astParent.astParent //跳过该不需要的控制节点
                      break()
                    }else{
                      //                    println("*****************")
                      q_parent = q_parent.astParent
                      break()
                    }
                    //                  if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else")){
                    //                    q_parent = q_parent.astParent.astParent //跳过该不需要的控制节点
                    //                  }else{
                    //                    q_parent = q_parent.astParent
                    //                    break()
                    //                  }
                  }
                }
              }
              else if(q_parent.code.contains("else")){
                //              println("elseelseelse")
                find_result += q_parent
                q_parent = q_parent.astParent //根据结构找到else的直接父节点if
                //              println("q_parent_jump _if",q_parent)
                if(q_parent != same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
                  breakable{
                    while(q_parent.code.contains("if")){
                      if(q_parent.astParent.astParent.isControlStructure && q_parent.astParent.astParent.code.contains("else")){
                        //并且else节点的子节点的个数只有一个（目前只能这样子）
                        //                      println("herehere",q_parent.astParent.astParent)
                        //                      println(q_parent.astParent.astParent.astChildren.l)
                        if(q_parent.astParent.astParent.astChildren.astChildren.l.length == 1){
                          //                        println("1111111")
                          q_parent = q_parent.astParent.astParent.astParent //跳过该不需要的控制节点
                          break()
                        }else{
                          //                        println("lenth>1")
                          q_parent = q_parent.astParent
                          //                        println("q_parent",q_parent)
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
              else if(q_parent.code.contains("switch")){//针对switch的情况
                //switch反到不用像if那样复杂，因为switch的每个分支的二级父节点直接是switch这个control节点
                /*这种情况是
                  2 switch外到switch内的判断语句这种情况要排除（？？？）
                  3 switch外到switch内的分支语句
                 */
                //println("find_switch")
                if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
                  //                println("find_case111")
                  //                println(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2)
                  find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
                }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
                  //                println("find_default")
                  find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
                }else{//普通变量入switch的判断语句情况
                  //                println("putomngruswitch")
                  find_result += q_parent
                }
                //              find_result += q_parent
                if(q_parent.astParent.id > cpg.method.l(1).id){
                  q_parent = q_parent.astParent
                }else{
                  break
                }
              }
            }
            else{
              //            println("q_parnet_front",q_parent)
              //            println(q_parent.astParent.id, cpg.method.l(1).id)
              //            if(q_parent.astParent.id > cpg.method.l(1).id) {
              if(q_parent.astParent.id > cpg.method.l(1).id){
                q_parent = q_parent.astParent
              }else{
                break
              }
              //            }
              //            println("查看一下应该等于啥",q_parent)
            }
            //          println("q_parent_last",q_parent)
//            println("q_parent",q_parent)
          }
        }


        //应对的是switch的情况
        if(q_parent == same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i)))){
          //println("am i here")
          if(q_parent.astParent.code.contains("switch")){
            //println("am i here1",q_parent.astParent.code,same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))))
            //应对的是同一个switch的不同分支的情况
            //思路：如果两个变量之间有case或default说明下一同名变量是swicth某一分支的入变量
            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
            }

          }else if(q_parent.isControlStructure && q_parent.code.contains("switch")){
            //println("am i here2")
            //应对的是switch的判断语句和不同分支
            //思路，也是寻找离下一同名变量节点最近的case和default
            //println("16,21")
            if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._1){
              //println("16,21",find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case"))
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"case")._2
            }else if(find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._1){
              find_result = find_result ++ find_some_node(list_identifier_use(i),list_identifier_use(next_same_name(i)),"default")._2
            }
          }
        }


        //下面的思路不太对吧
        //        if(q_parent == same_maxid_parent(list_identifier_use(i),list_identifier_use(next_same_name(i))) && q_parent.isControlStructure){
        //          //因为前有一个退出判断的条件是当判断到两个变量的公共父节点时，像while和for都有可能是中间没有同名变量，寻找中间的控制节点只有自己
        //          find_result += q_parent
        //        }
        //println("find_between_control",find_result)
        find_result
      }


      /*
      针对下一同名变量是普通变量的情况，由于有可能是入块的变量（普通变量也有可能是入块变量），并根据当前变量的类型有不同的处理
      函数对三种状态的判断是关于下一同名变量是否更新数据流的情况做的区分
      调用：common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1,next_is_out_control(0)._1,next_is_out_control(0)._2)
       */
      def common_yy_yn(i:Int,this_is_judge:String,this_is_out_control:String):Unit = {
        //根据下一同名变量的状态分情况讨论，但是需要for处理特殊的情况
        //有三连三，没三连二
        if(! update_data(i)._1){
          if(update_data(i)._2 == 2){
            //x=x+y
            //根据当前变量的情况，来选择下一同名变量的连边
//            println("common-2")
            if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")){
              //连边操作yy，当前变量是普通变量，所以从map_newuse取最新数据流
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
              //为可能的入块分别保存入块前数据流，当前变量为出结构体变量，下一同名变量的数据流来源应为出的块的数据流（map_last_stack）,应该处理完下一同名变量连边之后，再将map_last_stack弹出栈
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
//            println("common-3-4")
            //3（y=x）和4（函数参数）都在这里
//            println("++here",update_data(i)._2)
            if(update_data(i)._2 == 3){ //为自增这种操作加边 x+=3
              builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
            }
            if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")) {
//              println("in here")
              //连边操作yy，当前变量是普通变量，所以从map_newuse取最新数据流
              draw_line(i, list_identifier_use(next_same_name(i)))
//              println("draw line here")
//              println("find_between_control(i)",find_between_control(i))
              for (control_node <- 0 to find_between_control(i).length - 1) {
//                println("here")
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
//                println("map_last_stack",map_last_stack(list_identifier_use(i).code))
                //当前变量即为判断语句又为出块语句，不需要连边

                for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                  list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
                  builder.addEdge(list_identifier_use(next_same_name(i)),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                }

                for (control_node <- 0 to find_between_control(i).length - 1 ) {
//                  println("control_node",control_node)
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
//          println("common-1")
          //数据流被刷新，不需要连边，但是需要入map_newuse_stack，因为要为后面的分支考虑，并且需要更新map_newuse数据流
          if((this_is_judge == "false" && this_is_out_control == "false")||(this_is_judge != "false" && this_is_out_control == "false")) {
//            println("isisiisis")
//            println(find_between_control(i))
            for (control_node <- 0 to find_between_control(i).length - 1 ) {
              //println("is_in_here")
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
//              println("map_last",map_last_stack(list_identifier_use(i).code))
              for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
              }

              for (control_node <- 0 to find_between_control(i).length - 1 ) {
//                println("find_between_control",find_between_control(i)(find_between_control(i).length - 1 - control_node))
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  if(next_same_name(next_same_name(i)) != -1){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }else{
                  if(next_same_name(next_same_name(i)) != -1){
//                    println("list_identifier_use(next_same_name(next_same_name(i)))",list_identifier_use(next_same_name(next_same_name(i))))
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
        if(next_is_out_control != "false"){//代表下一同名变量是出结构体变量
          if(next_is_out_control_node.code.contains("for") && next_is_out_control_node.code.startsWith("for")){//如果是for需要对下一同名变量为出结构体的内部变量做连边处理
            if(map_For_Inner.get(next_is_out_control_node) != None){//判断是否为当前for块的内部变量
              if(map_For_Inner(next_is_out_control_node).contains(list_identifier_use(i).code)){
                is_inner = true
              }
            }
            if(is_inner){//如果是for的内部变量，则进行连边处理，有3连3，没有3连2
              if(map_For_Outer.get(next_is_out_control_node) != None){
                if(map_For_Outer(next_is_out_control_node).get(list_identifier_use(i).code) != None){
                  for(k <- 0 to map_For_Outer(next_is_out_control_node)(list_identifier_use(i).code).length-1){
//                    println("不为空",map_For_Outer(next_is_out_control_node)(list_identifier_use(i).code)(k))
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
        //__ny和__nn的情况都是下一同名变量为判断语句，因此需要保存入块前数据流和连边以及更新map_newuse
        if(this_is_judge == "false" && this_is_out_control == "false"){
          //yyny 和yynn
          draw_line(i, list_identifier_use(next_same_name(i)))
//          println("isis")
//          println("find_between_control(i)",find_between_control(i))
          //如果当前入的结构体为for，第二分支和第三分支的变量都需要入map_newuse_stack,因为for块出变量还会影响到他们
          for (control_node <- 0 to find_between_control(i).length - 1) {
            //处理下一同名变量为for的入块变量
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
                  //map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node)).put(list_identifier_use(i).code,ListBuffer(list_identifier_use(next_same_name(i))))
                }
                //对于第一个来说只需要连边，数据流不要更新（因为若有2，3分支需要为其保留入for前数据流）,
                // 虽然为入块变量但后续不参与出块相关，所以不保存？这样如果没有二三分支，当前变量为第一分支变量，下一同名变量为入块时也可以
                draw_line(i,list_identifier_use(next_same_name(i)))
              }else if(for_belong_which == 2 || for_belong_which == 3){
                if(for_belong_which == 2){
//                  println("into for 2")
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
//                  println("into for 3")
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
//                  println("map_For_Outer",map_For_Outer)
                }
                //连边并更新map_newuse_stack保存入块前数据流
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
            else{ //目前是if和while
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
          //ynny ynnn nnny nnnn
          var list_out_data = new ListBuffer[AstNode]
          if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && ! map_last_stack(list_identifier_use(i).code).isEmpty){
            for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
              list_out_data += map_last_stack(list_identifier_use(i).code).head(k)._2
              builder.addEdge(list_identifier_use(next_same_name(i)),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
            }
            for (control_node <- 0 to find_between_control(i).length - 1) {
              //处理下一同名变量为for的入块变量¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
              if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for") && find_between_control(i)(find_between_control(i).length - 1 - control_node).code.startsWith("for")){
                var for_belong_which = for_Belong_Which(list_identifier_use(next_same_name(i)),find_between_control(i)(find_between_control(i).length - 1 - control_node))
                if(for_belong_which == 1){
                  //对于第一个来说只需要连边，数据流不要更新（因为若有2，3分支需要为其保留入for前数据流）,
                  // 虽然为入块变量但后续不参与出块相关，所以不保存？这样如果没有二三分支，当前变量为第一分支变量，下一同名变量为入块时也可以
                  draw_line(i,list_identifier_use(next_same_name(i)))
                }else if(for_belong_which == 2 || for_belong_which == 3){
                  //连边并更新map_newuse_stack保存入块前数据流,入块前数据流可能不只一个
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
              else{//if和while
                if(map_newuse_stack.isEmpty || map_newuse_stack.get(list_identifier_use(i).code) == None || map_newuse_stack(list_identifier_use(i).code).isEmpty){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_out_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                }else{
//                  println("why",map_newuse_stack(list_identifier_use(i).code),find_between_control(i)(find_between_control(i).length - 1 - control_node),list_out_data,ListBuffer(list_identifier_use(next_same_name(i))))
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
        //下一同名变量虽然为普通语句，但也有可能是入块（可能为多个）的变量，所以需要进行判断并通过判断后为块保存入块前数据流
        if(! update_data(i)._1){ //根据下一同名变量是否更新数据流分类讨论
          if(update_data(i)._2 == 2){//x=x+y
            if(this_is_judge != "false" && this_is_out_control == "false"){//ny__
              /*
              当前变量为判断语句，下一同名变量为普通变量的情况：（while和for应该是只有这种情况（即当前变量是判断语句））
              1 将下一同名变量与判断语句中所有的同名变量连边
              2 判断为普通变量的下一同名变量是否为其他块的入块变量（使用find_between_control），若有，入栈、连边
               */
              //与yn__和nn__的区别：
              //1
              //此时map_newuse_stack栈顶保存的肯定是当前变量作为判断语句所属的块
              if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                //for的判断语句有多种情况，所以不太一样，所以连边要分情况讨论
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
                else{ //if while
                  for(node <- map_newuse_stack(list_identifier_use(i).code).head._3){
                    builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),node,"LastUse")
                  }
                }
              }else{ //有可能是判断语句是代码中第一个出现该名称的变量（一般代码正确的话不会出现这种情况），那么就将当前变量自己作为判断语句变量，不考虑判断语句是否有其他同名变量
                builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),list_identifier_use(i),"LastUse")
              }
              //2 并且若普通变量为入块变量,保存的入块前数据流就是当前变量（也就是判断语句变量）
              for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
//                //如果是for块的判断语句到块的普通变量
//                if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for")){
//                  var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),find_between_control(i)(find_between_control(i).length - 1 - control_node))
//                  /*
//                  如果for_Belong_This == 1 ，代表只有第一个分支，那么就由第一个分支连边
//                  如果for_Belong_This == 2 ，代表第二个分支存在，只要第二个分支存在就使用第二个分支的变量连接（即使第一个分支也存在）
//                  如果for_Belong_This == 3 ，代表第三个分支存在，无论前两个分支存不存在，都需要下一同名变量连向第三个分支，
//                      但是需要判断，如果第二个分支存在（也可以是一二分支都存在），那么下一同名变量需要连向第二个分支的变量，
//                                  如果第二个分支不存在，判断第一个分支是否存在，如果存在（一存在，二不存在），增加连接向第一个分支的边
//                  所以无论for_Belong_This的取值，都需要下一同名变量向上连边
//                   */
//                  if(for_Belong_This == 3){
//                    if(map_For_Second.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code)!= None){
//                      for(k <- 0 to map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
//                        builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
//                      }
//                    }else{
//                      if(map_For_First.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
//                        for(k <- 0 to map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
//                          builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))),map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
//                        }
//                      }
//                    }
//                  }
//                }
                //压入栈，并且连边（无论下一同名变量是不是入块变量，前面的连边已经解决了这个地方连边的需求，因为即使因为下一同名变量为入块变量与入块前数据流连边，和前面当前变量与下一同名变量连边是相同的）
                //if和while和for
                if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                  map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                }else{//当前变量代表的判断语句所在的判断条件里的所有同名变量作为下一同名变量可能为入块变量时的入块数据流
                  /*
                  也需要分情况讨论：
                  若当前map_newuse_stack栈顶保存的._1块节点不是当前变量所在的块节点，说明栈顶保存的是上一轮for循环中，内部其他块也将下一同名变量作为入块变量的情况
                  */
                  if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_judge_node){
                    //当前栈顶保存的是当前变量所属的判断语句的块
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }else{
                    //若不是，入块钱数据流应该与当前栈顶的入块前数据流一样
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._2
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                  }
                }
              }
            }
            else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
              //由于当前变量为出分支变量，下一同名变量在新的分支中，所以新的分支需要继承入数据流，也就是压入栈中
              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                //由于该种情况下为if块的出一个分支，入一个分支的情况，order_or_nest只需要考虑if的情况
                //为当前出分支保存出分支数据流，并到整个if块里的出分支
                if(find_between_control(i).length > 0){
                  if (order_or_nest_plus(map_newuse_stack(list_identifier_use(i).code).head._1, find_between_control(i).last) == 5) {
                    for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
                      map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_data, ListBuffer(list_identifier_use(next_same_name(next_same_name(i))))))
                    }
                  }
                  //并且该种情况下，下一同名变量不为判断语句，因此为某一分支的入数据流，因此，要为该分支的如数据流与入if块前的数据流连边
                  for (p <- 0 to list_data.length - 1) {
                    builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), list_data(p), "LastUse")
                  }
                }else{
                  draw_line(i,list_identifier_use(next_same_name(next_same_name(i))))
                }
              }
            }
            //无论什么情况都要更新数据流
            update_map_newuse(i,list_identifier_use(next_same_name(next_same_name(i))))
          }
          else{
            if(update_data(i)._2 == 3){ //为自增这种操作加边 x+=3
              builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
            }
            if(this_is_judge != "false" && this_is_out_control == "false"){
              //2 并且若普通变量为入块变量保存的入块钱数据流就是当前变量（也就是判断语句变量）
              if(find_between_control(i).length > 1){
                for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
                  if(find_between_control(i)(find_between_control(i).length - 1 - control_node).code.contains("for") && find_between_control(i)(find_between_control(i).length - 1 - control_node).code.startsWith("for")){
                    //println("is_here")
                    var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),find_between_control(i)(find_between_control(i).length - 1 - control_node))
                    if(for_Belong_This == 3){
                      //println("map_For_Second",map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node)))
                      if(map_For_Second.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                        //println("2 不为空")
                        for(k <- 0 to map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                          builder.addEdge(list_identifier_use(next_same_name(i)),map_For_Second(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                        }
                      }else{
                        //println("1 不为空",map_For_First.get(find_between_control(i)(find_between_control(i).length - 1 - control_node)))
                        if(map_For_First.get(find_between_control(i)(find_between_control(i).length - 1 - control_node))!= None && map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node)).get(list_identifier_use(i).code) != None){
                          //println("map_For_First",map_For_First)
                          for(k <- 0 to map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                            builder.addEdge(list_identifier_use(next_same_name(i)),map_For_First(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                          }
                        }
                      }
                      //println("map_For_Outer",map_For_Outer)
                      //第三个分支也会影响入块的第一个变量，需要将入块变量连向第三分支变量（无论是for内部变量还是外部变量）
                      for(k <- 0 to map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code).length - 1){
                        builder.addEdge(list_identifier_use(next_same_name(i)),map_For_Outer(find_between_control(i)(find_between_control(i).length - 1 - control_node))(list_identifier_use(i).code)(k),"LastUse")
                      }
                    }
                    else{
                      builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(i),"LastUse")
                    }
                  }
                  //if while for
                  if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                    map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                  }else{//当前变量代表的判断语句所在的判断条件里的所有同名变量作为下一同名变量可能为入块变量时的入块数据流
                    var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                    map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(i)))))
                  }
                }
              }
              else{
//                println("here")
                //下一同名变量不是入结构体变量
                //入块变量与当前所在块的判断语句连边
//                if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                  for(line <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse_stack(list_identifier_use(i).code).head._3(line),"LastUse")
                  }
                }
              }

              //无论何种情况都需要更新数据流
              update_map_newuse(i,list_identifier_use(next_same_name(i)))
            }
            else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                //与上一种情况同理，只应用于if
                if(find_between_control(i).length > 0){
                  if (order_or_nest_plus(map_newuse_stack(list_identifier_use(i).code).head._1, find_between_control(i).last) == 5) {
                    //由于是分支关系，入栈之前把上一分支的弹出（因为上一分支彻底出了）
                    map_newuse_stack(list_identifier_use(i).code).pop()
                    // 为所有通过该变量入块的块记录入块前数据流
                    for (control_node <- 0 to find_between_control(i).length - 1) {
                      map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node), list_data, ListBuffer(list_identifier_use(next_same_name(i)))))
                    }
                  }
                  //若普通变量为入块变量，那么入栈后，也要将该入块变量连向入块数据流
                  for (p <- 0 to list_data.length - 1) { //为当前判断语句连边
                    builder.addEdge(list_identifier_use(next_same_name(i)), list_data(p), "LastUse")
                  }
                }
                else{
                  //如果下一同名变量为普通变量，即不为隐性或显性的入块变量，那么需要为下一同名变量与当前变量连边
                  draw_line(i,list_identifier_use(next_same_name(i)))
                }
              }
            }
            update_map_newuse(i,list_identifier_use(next_same_name(i)))
            //println("map_newuse",map_newuse(list_identifier_use(i).code))
          }
        }
        else{
          if(this_is_judge != "false" && this_is_out_control == "false"){
            //builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(i),"LastUse")
            //2 并且若普通变量为入块变量保存的入块钱数据流就是当前变量（也就是判断语句变量）
            for (control_node <- 0 to find_between_control(i).length - 1 if find_between_control(i).length > 1) {
              if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                map_newuse_stack(list_identifier_use(i).code) = Stack((find_between_control(i)(find_between_control(i).length - 1 - control_node), ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
              }else{//当前变量代表的判断语句所在的判断条件里的所有同名变量作为下一同名变量可能为入块变量时的入块数据流
                var list_tmp = map_newuse_stack(list_identifier_use(i).code).head._3
                map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - control_node),list_tmp,ListBuffer(list_identifier_use(next_same_name(i)))))
              }
            }
          }
          else if((this_is_judge == "false" && this_is_out_control != "false")||(this_is_judge != "false" && this_is_out_control != "false")){
            //数据流被刷新，不需要连边，但是需要入map_newuse_stack，因为要为后面的分支考虑，并且需要更新map_newuse数据流
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {

//              println("为什么取不到",map_newuse_stack(list_identifier_use(i).code))
              if(find_between_control(i).length > 0){
                var list_data: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2
                //与前两种情况同理，order_or_nest只应用于
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
          //该种情况下，下一同名变量即使是入块变量，肯定与当前变量所在块为嵌套关系
          if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty) {
            var newuse_3: ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._3 //入块前数据流
            //map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2, newuse_3, ListBuffer(list_identifier_use(next_same_name(i)))))
            //连边
            for (p <- 0 to newuse_3.length - 1) {
              builder.addEdge(list_identifier_use(next_same_name(i)), newuse_3(p), "LastUse")
            }
            //再解决内层循环的问题
            for (q <- 0 to find_between_control(i).length - 1) {//从外层开始判断，因为考虑了map_newsue_stack入栈的顺序
//              println("(find_between_control(i).length - 1 - q)",find_between_control(i)(find_between_control(i).length - 1 - q))

              //只能是嵌套关系，因为若是分支关系，那么当前变量一定既是判断语句也是出块语句
              if(order_or_nest_plus(this_is_judge._2,find_between_control(i)(find_between_control(i).length - 1 - q)) == 1){
                map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - q), newuse_3, ListBuffer(list_identifier_use(next_same_name(i)))))

              }
//              if (order_or_nest_plus(find_between_control(i)(find_between_control(i).length - 1 - q), this_is_judge._2) == 1) { //嵌套关系
//                map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length - 1 - q), newuse_3, ListBuffer(list_identifier_use(next_same_name(i)))))
//              }
            }
          }
        }
        else if((this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false")||(this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false")){
//          println("test_last_stack",map_last_stack(list_identifier_use(i).code))
          if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && !map_newuse_stack(list_identifier_use(i).code).isEmpty){
//          if(map_newuse_stack.get(list_identifier_use(i).code) != None){

            //在出结构体直接pop map_newuse_stack,入块数据流无法继承

//            println("map_newuse_stack",map_newuse_stack(list_identifier_use(i).code))
            var newuse_2:ListBuffer[AstNode] = map_newuse_stack(list_identifier_use(i).code).head._2//入块前数据流
            //保留完入块数据流再弹出栈
            map_newuse_stack(list_identifier_use(i).code).pop()
//            println("newuse_2",newuse_2)
            /*
              next_is_judge有3种情况：并且这三种情况this_is_out_control.last._2的id小于next_is_judge
              1 next_is_judge是嵌套在this_is_out_control.last._2内的块
              不存在该种情况，因为当前变量已经是出结构体的变量了
              2 next_is_judge是this_is_out_control.last._2的下行顺序if分支
              3 next_is_judge是this_is_out_control.last._2的下行顺序if分支中嵌套的块
             */

            //把这两种情况写出来，再分别测试一下，然后
//            if(order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2) == 1){
//
//            }
//            println("this_is_out_control.last._2,next_is_judge._2",this_is_out_control.last._2,next_is_judge._2)
//            println("order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2)",this_is_out_control.last._2,next_is_judge._2,order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2))
            if(order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2) == 5){
              //顺序关系，说明为同一块的分支,继承入块前数据流(update map_newuse_stack)
              //不需要弹出栈，因为弹出map_newuse_stack栈在当前变量出块的时候就操作了（out_control和out_control_common）
              //map_newuse_stack(list_identifier_use(i).code).pop()
              map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
              //连边
              for(p <- 0 to newuse_2.length-1){
                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
              }
            }
            else if(order_or_nest_plus(this_is_out_control.last._2,next_is_judge._2) == 4){
              //先为外层顺序分支入块，再由外向内为其他块入map_newuse_stack入栈
              //先解决最外层的if的入map_newuse_stack
              //map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
              //连边
              for(p <- 0 to newuse_2.length-1){
                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
              }
              //再解决内层循环的问题
//              println("find_between_control(i)",find_between_control(i))
              for(q <- 0 to find_between_control(i).length-1) {
                //如果结构体节点与this_is_judge._2是顺序关系，则认为是无关，若是嵌套关系则该外层块的入块变量也是当前下一同名变量
                //从外层开始判断，因为考虑了map_newsue_stack入栈的顺序
//                println("find_between_control(i)",find_between_control(i)(find_between_control(i).length-2-q))
//                println("find_between_control(i)",find_between_control(i)(find_between_control(i).length-1-q))
                if(find_between_control(i)(find_between_control(i).length-1-q) == next_is_judge._2){
//                  println("相同")
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length-1-q),newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
                }
//                println("order_or_nest_plus(find_between_control(i)(find_between_control(i).length-1-q),next_is_judge._2)",find_between_control(i)(find_between_control(i).length-1-q),next_is_judge._2,order_or_nest_plus(find_between_control(i)(find_between_control(i).length-1-q),next_is_judge._2))
                if(order_or_nest_plus(find_between_control(i)(find_between_control(i).length-1-q),next_is_judge._2) == 1) {//嵌套关系
//                  println("is_sd")
                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length-1-q),newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
                }
              }
            }
//            if(order_or_nest(this_is_out_control.last._2,next_is_judge._2) && (next_is_judge._1 == "else" || next_is_judge._1 == "if")) {//如果是if块的顺序分支关系
//              //还是判断顺序和分支有问题order_or_nest
//              println("是else还是while，是else")
//              //如果是顺序关系，说明为同一块的分支,继承入块前数据流(update map_newuse_stack)
//              map_newuse_stack(list_identifier_use(i).code).pop()
//              map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
//              //连边
//              for(p <- 0 to newuse_2.length-1){
//                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
//              }
//            }else {
//              //先解决最外层的if的map_newuse_stack更新问题
//              map_newuse_stack(list_identifier_use(i).code).pop()
//              map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
//              //连边
//              for(p <- 0 to newuse_2.length-1){
//                builder.addEdge(list_identifier_use(next_same_name(i)),newuse_2(p),"LastUse")
//              }
//              //再解决内层循环的问题
//              println("find_between_control(i)",find_between_control(i))
//              for(q <- 0 to find_between_control(i).length-2) {
//                //如果结构体节点与this_is_judge._2是顺序关系，则认为是无关，若是嵌套关系则该外层块的入块变量也是当前下一同名变量
//                //从外层开始判断，因为考虑了map_newsue_stack入栈的顺序
//                println("find_between_control(i)",find_between_control(i)(find_between_control(i).length-2-q))
//                if(!order_or_nest(find_between_control(i)(find_between_control(i).length-2-q),this_is_judge._2)) {//嵌套关系
//                  map_newuse_stack(list_identifier_use(i).code).push((find_between_control(i)(find_between_control(i).length-1-q),newuse_2,ListBuffer(list_identifier_use(next_same_name(i)))))
//                }
//              }
//            }
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
//        if(this_is_out_control.length > 1){
          //处理除了最外层的出分支情况，for循环处理的全部都是出完整的块
        for (node <- 0 to this_is_out_control.length-1){
//          println("this_is_out_control(node)._1",this_is_out_control(node)._1)
          //注意这里把最外层留下之后去处理，嵌套由内向外进行处理
          //除了最外层，当前出的所有块都在循环中处理结束，并且内层出块，即使是多分支也是整个多分支都出
          if(this_is_out_control(node)._1 == "while"){
//            println("out-while-if")
            //先判断while判断语句中是否有同名变量,即有没有判断语句向出结构体连边的需求
            //如果while的判断语句中有同名变量，那么连边，弹出map_newuse_stack，如果没有，既不需要连边也不需要弹出栈（无需任何操作）
            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                //如果判断语句中有同名变量
                /*如果当前map_lastuse_stack保存的栈顶节点是while节点的子节点，说明while内部有其他多分支的块，
                  即出数据流不仅只有当前变量，还有while内部多分支的块，通过看while节点与map_last_stack保存的控制节点的关系，
                  如果是父节点，那么就说明确实存在别的块，需要考虑连边，
                  连边之后map_last_stack需要继续保存，属于当前while的map_newuse_stack栈顶的元素需要弹出
                 */
                if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None && ! map_last_stack(list_identifier_use(i).code).isEmpty){
                  if(is_father(this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1)){
                    //连边，并将map_newuse_stack栈顶弹出
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
              //将map_newuse_stack栈顶保存的当前while相关弹出,无论while的判断语句中是否有当前出块变量的同名变量，都要弹出，因为入栈的时候两种情况都入栈了
              map_newuse_stack(list_identifier_use(i).code).pop()
            }
          }
          else if(this_is_out_control(node)._1 == "if" || this_is_out_control(node)._1 == "switch"){
            //目前与out_control_common的区别：弹出map_newuse_stack的区别，
            // 出if分支，只弹出内层，为下一分支保存入块数据流
            // 出整个块，全部弹出，不需要保存入块数据流
//            println("out-if-if")

            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
              //栈为空，说明该分支是其所在多分支唯一的出数据流分支
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
//                println("2.1")
                //2.1
                var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
                map_last_stack(list_identifier_use(i).code).pop()
                if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
                  //弹出栈后，栈可能为空，因为有可能当前栈里只有一个内层完全出块 if(){if(){x>0}}
                  //内层完全出栈，所以将栈顶保存的内层if的出数据流继承给当前正在判断的外层，并且由于内层有多个出数据流，因此需要循环从list_tmp取出，更换出结构体
                  var list_change_name = new ListBuffer[(AstNode,AstNode)]
                  for (list_node <- 0 to list_tmp.length-1){
                    list_change_name +=((this_is_out_control(node)._2,list_tmp(list_node)._2))
                  }
                  map_last_stack(list_identifier_use(i).code) = Stack(list_change_name)
                }
                else{
                  if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                    //判断此时的栈顶与当前出分支是顺序分支关系
                    var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                    map_last_stack(list_identifier_use(i).code).pop()
                    map_last_stack(list_identifier_use(i).code).push(list_concat)
                  }else{
                    //可能当前分支的其他同级分支中没有出数据流，那么就将内部的出块再压入map_last_stack栈顶，没有必要替换控制节点，因为相对于外层来说都是嵌套的关系
                    map_last_stack(list_identifier_use(i).code).push(list_tmp)
                  }
                }
              }
              else{
//                println("2.2")
                var list_tmp = new ListBuffer[(AstNode,AstNode)]
                if(map_newuse.get(list_identifier_use(i).code) != None){
                  list_tmp= ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head))
                }
                else{
                  list_tmp = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i)))
                }
                //2.2
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                  //如果当前栈顶保存的出数据流是与当前出分支属于同一if，即当前出的if块其他分支也有出数据流
                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)
                }
                else{
                  //内部没有其他出块分支，并且同级的其他分支也没有出块数据流，因此只为当前出分支保存出数据流作为整个if块的出数据流
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
//              println("out_if_map_newuse_stack",map_newuse_stack(list_identifier_use(i).code))
            }else{
//              println("out_if_map_newuse_stack,empty")
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
//              println("out_if_map_last_stack",map_last_stack(list_identifier_use(i).code))
            }else{
//              println("out_if_map_last_stack,empty")
            }
          }
          else if(this_is_out_control(node)._1 == "for"){
            //非for内部变量的情况，当前变量为for的

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
//          println("all out nodes:",this_is_out_control(node))
          //由内向外进行处理嵌套
          //除了最外层，当前出的所有块都在循环中处理结束
          if(this_is_out_control(node)._1 == "while"){
            //与out_control处理while时的操作完全相同
//            println("out_control_common_while")
            //保存出while的数据流，为出块与下一变量连边
            //整理出数据流
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
              //map_last_stack为空，就把当前的出数据入map_last_stack
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
                //1
                if(order_or_nest_plus(this_is_out_control(node)._2, map_last_stack(list_identifier_use(i).code).head(0)._1) == 1){
                  //栈顶保存的是while内部嵌套的其他块的出变量,在栈顶的列表加入while的出数据流
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
                //2
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 4){
                  //栈顶保存的是其他块的出块变量，并且当前块是多分支块，还没有完全出块
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
                  }else{
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
                  }
                }
              }
            }

            //针对while是个循环体，出块变量会影响到入块变量（无论是判断语句还是作为入块的普通变量），因此需要出块变量与入块变量相连
            //因为即使判断语句没有同名变量，也会有普通变量作为入块的第一个变量，该变量也会受出块变量的影响，所以只需要判断map_newuse_stack当前栈顶._1保存的是不是当前出的while块节点
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){

//              if(map_newuse_stack.get(list_identifier_use(i).code) != None){
              //println("map_newuse_stack(list_identifier_use(i).code).head._1",map_newuse_stack(list_identifier_use(i).code))
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                //println("here-in")
                //在上面把map_last_stack处理好了，所以这个地方不需要担心map_last_stack栈顶保存的不是当前while块的
                if(map_last_stack.get(list_identifier_use(i).code) != None){
                  //println("this_is_out_control(node)._2",this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1)
                  if(this_is_out_control(node)._2 == map_last_stack(list_identifier_use(i).code).head(0)._1){
                    //连边，并将map_newuse_stack栈顶弹出
                    for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                      for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                        builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m), map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                      }
                    }
                  }
                }
                else{
                  //这种情况可能都不存在
                  for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    if(map_newuse.get(list_identifier_use(i).code) == None){
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),list_identifier_use(i),"LastUse")
                    }else{
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }
                  }
                }
                //将map_newuse_stack栈顶保存的当前while相关弹出
                map_newuse_stack(list_identifier_use(i).code).pop()
              }
            }
            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
//              println("out_while_map_newuse_stack",map_newuse_stack(list_identifier_use(i).code))
            }else{
//              println("out_while_map_newuse_stack,empty")
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
//              println("out_while_map_last_stack",map_last_stack(list_identifier_use(i).code))
            }else{
//              println("out_while_map_last_stack,empty")
            }
          }
          else if(this_is_out_control(node)._1 == "if" || this_is_out_control(node)._1 == "switch"){
//            println("out-whole-if")
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
              //栈为空，说明该分支是其所在多分支唯一的出数据流分支
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
//                println("2.1")
                //2.1
                var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
                map_last_stack(list_identifier_use(i).code).pop()
                if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
                  //弹出栈后，栈可能为空，因为有可能当前栈里只有一个内层完全出块 if(){if(){x>0}}
                  //内层完全出栈，所以将栈顶保存的内层if的出数据流继承给当前正在判断的外层，并且由于内层有多个出数据流，因此需要循环从list_tmp取出，更换出结构体
                  var list_change_name = new ListBuffer[(AstNode,AstNode)]
                  for (list_node <- 0 to list_tmp.length-1){
                    list_change_name +=((this_is_out_control(node)._2,list_tmp(list_node)._2))
                  }
                  map_last_stack(list_identifier_use(i).code) = Stack(list_change_name)
                }
                else{
                  if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                    //判断此时的栈顶与当前出分支是顺序分支关系
                    var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                    map_last_stack(list_identifier_use(i).code).pop()
                    map_last_stack(list_identifier_use(i).code).push(list_concat)
                  }else{
                    //可能当前分支的其他同级分支中没有出数据流，那么就将内部的出块再压入map_last_stack栈顶，没有必要替换控制节点，因为相对于外层来说都是嵌套的关系
                    map_last_stack(list_identifier_use(i).code).push(list_tmp)
                  }
                }
              }
              else{
//                println("2.2")
                var list_tmp = new ListBuffer[(AstNode,AstNode)]
                if(map_newuse.get(list_identifier_use(i).code) != None){
                  list_tmp= ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head))
                }
                else{
                  list_tmp = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i)))
                }
                //2.2
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
                  //如果当前栈顶保存的出数据流是与当前出分支属于同一if，即当前出的if块其他分支也有出数据流
                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
                  map_last_stack(list_identifier_use(i).code).pop()
                  map_last_stack(list_identifier_use(i).code).push(list_concat)
                }
                else{
                  //内部没有其他出块分支，并且同级的其他分支也没有出块数据流，因此只为当前出分支保存出数据流作为整个if块的出数据流
                  map_last_stack(list_identifier_use(i).code).push(list_tmp)
                }
              }
            }

//            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
//              //栈为空，说明该分支是其所在多分支唯一的出数据流分支
//              map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
//            }
//            else{
//              /*
//              分为两种情况
//              1 当前出的if是内层if，并且肯定该内层if也是完全出结构体，所以需要将栈顶保存的内层if的出数据流合并到外层（可能是多种块）的出数据流里
//              2 当前出的if是出的最外层的整个多分支if，但是也分为两种情况
//                2.1 该外层if分支的内部有其他出块，所以当前map_last_stack栈顶保存的是内部出块的出数据流，(map_last_stack栈顶保存的节点id大于当前出分支节点id)
//                2.2 该外层if分支的内部没有其他出块，直接将当前出分支数据流并入map_last_stack栈顶，()
//               */
//              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
//                println("2.1")
//                //2.1
//                var list_tmp:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head
//                map_last_stack(list_identifier_use(i).code).pop()
//                println(map_last_stack(list_identifier_use(i).code))
//                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
//                  //判断此时的栈顶与当前出分支是顺序分支关系
//                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
//                  map_last_stack(list_identifier_use(i).code).pop()
//                  map_last_stack(list_identifier_use(i).code).push(list_concat)
//                }else{
//                  //可能当前分支的其他同级分支中没有出数据流，那么就将内部的出块再压入map_last_stack栈顶，没有必要替换控制节点，因为相对于外层来说都是嵌套的关系
//                  map_last_stack(list_identifier_use(i).code).push(list_tmp)
//                }
//              }
//              else{
//                println("2.2")
//                var list_tmp = new ListBuffer[(AstNode,AstNode)]
//                if(map_newuse.get(list_identifier_use(i).code) != None){
//                  list_tmp= ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head))
//                }
//                else{
//                  list_tmp = ListBuffer((this_is_out_control(node)._2,list_identifier_use(i)))
//                }
//                //2.2
//                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 5){
//                  //如果当前栈顶保存的出数据流是与当前出分支属于同一if，即当前出的if块其他分支也有出数据流
//                  var list_concat:ListBuffer[(AstNode,AstNode)] = map_last_stack(list_identifier_use(i).code).head ++ list_tmp
//                  map_last_stack(list_identifier_use(i).code).pop()
//                  map_last_stack(list_identifier_use(i).code).push(list_concat)
//                }
//                else{
//                  //内部没有其他出块分支，并且同级的其他分支也没有出块数据流，因此只为当前出分支保存出数据流作为整个if块的出数据流
//                  map_last_stack(list_identifier_use(i).code).push(list_tmp)
//                }
//              }
//            }
            //因为出了当前的if块，若if块有入结构体变量，需要从map_newuse_stack弹出
            //而且我们认为普通变量也有可能是入结构体变量，因此map_newuse_stack是一定存在的
            //？？？？？？目前存在问题？？？？？ 某个变量首次出现可能是入结构体变量，但是入块时，是下一同名变量入块时处理，导致当前变量没有入map_newuse_stack

//            if((!map_newuse_stack.isEmpty) &&  (!map_last_stack(list_identifier_use(i).code).isEmpty)){
//              println("kong",map_newuse_stack.isEmpty,map_newuse_stack.get(list_identifier_use(i).code) ,map_last_stack(list_identifier_use(i).code).isEmpty)
//              println(map_newuse_stack(list_identifier_use(i).code))
//              map_newuse_stack(list_identifier_use(i).code).pop()
//            }
//            val containsEmptyTuple : Boolean = map_newuse_stack(list_identifier_use(i).code).headOption match {
//              case Some(()) => true
//              case _ => false
//            }
//            if(!map_newuse_stack.isEmpty && !containsEmptyTuple){
//              map_newuse_stack(list_identifier_use(i).code).pop()
//            }
            if(!map_newuse_stack.isEmpty){
              if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                val containsEmptyTuple: Boolean = map_newuse_stack(list_identifier_use(i).code).headOption.exists(_ == (()))

                if(!containsEmptyTuple){
//                  println("containsEmptyTuple",containsEmptyTuple)
//                  println(!map_newuse_stack(list_identifier_use(i).code).isEmpty)
                  if(map_last_stack(list_identifier_use(i).code).isEmpty){
//                    print("here?")
                    map_newuse_stack(list_identifier_use(i).code).pop()
                  }


                }
              }

            }

//            if((!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None)&&(!map_last_stack(list_identifier_use(i).code).isEmpty)){
//              println("new_newuse_stack(list_identifier_use(i).code)",map_newuse_stack(list_identifier_use(i).code))
//              map_newuse_stack(list_identifier_use(i).code).pop()
//            }

            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
//              println("out_if_map_newuse_stack",map_newuse_stack(list_identifier_use(i).code))
            }else{
//              println("out_if_map_newuse_stack,empty")
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
//              println("out_if_map_last_stack",map_last_stack(list_identifier_use(i).code))
            }else{
//              println("out_if_map_last_stack,empty")
            }
          }
          else if(this_is_out_control(node)._1 == "for"){

          }
          else if(this_is_out_control(node)._1 == "do-while"){
            //当前变量为出do-while的情况
//            println("out-whole-do-while")
            /*
            while的判断语句和出do块（while中没有同名变量）作为出结构体的变量，都会进入这里
            需要做的操作：
            1 将出块变量入map_last_stack栈，和while一样，也要根据map_last_stack栈顶保存的元素分情况讨论
            2 和while的情况一样，因为是个循环体，出块变量会影响到入块变量，（while语句出块和do块普通变量作为出块是一样的）
             */
            if(map_last_stack.isEmpty || map_last_stack.get(list_identifier_use(i).code) == None || map_last_stack(list_identifier_use(i).code).isEmpty){
              //println("MMM",map_last_stack.isEmpty,map_last_stack.get(list_identifier_use(i).code) == None)
              if(map_newuse.get(list_identifier_use(i).code) != None){
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
              }else{
                map_last_stack(list_identifier_use(i).code) = Stack(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
              }
            }
            else{
              //println("???",map_last_stack(list_identifier_use(i).code).isEmpty,map_last_stack.get(list_identifier_use(i).code).isEmpty)

              /*
              不为空，可能当前栈顶不是当前判断的出块的节点
              有两种情况，与while的情况完全相同
              1 栈顶保存的是while内部嵌套的其他块的出变量，有可能是多分支的情况，需要将栈顶合并进当前while块的出数据流
              2 栈顶保存的是其他块的出块变量，并且当前块是多分支块，还没有完全出块
                if(){x}else if(){while(){x}} 此时map_last_stack栈顶保存的就是if第一个分支的出数据流
               */

              //判断当前栈顶的节点与当前出块节点的关系，
              if(map_last_stack(list_identifier_use(i).code).head(0)._1.id > this_is_out_control(node)._2.id){
                if(order_or_nest_plus(this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1) == 1){
                  //栈顶保存的是while内部嵌套的其他块的出变量,在栈顶的列表加入while的出数据流
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
                //println("1111111",order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2))
                if(order_or_nest_plus(map_last_stack(list_identifier_use(i).code).head(0)._1,this_is_out_control(node)._2) == 4){
                  //println("2222222")
                  //栈顶保存的是其他块的出块变量，并且当前块是多分支块，还没有完全出块
                  if(map_newuse.get(list_identifier_use(i).code) != None){
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,map_newuse(list_identifier_use(i).code).head)))
                  }else{
                    map_last_stack(list_identifier_use(i).code).push(ListBuffer((this_is_out_control(node)._2,list_identifier_use(i))))
                  }
                }
              }
            }

            if(map_newuse_stack.get(list_identifier_use(i).code) != None){
              //println("map_newuse_stack(list_identifier_use(i).code).head._1",map_newuse_stack(list_identifier_use(i).code))
              if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2){
                //println("here-in")
                //在上面把map_last_stack处理好了，所以这个地方不需要担心map_last_stack栈顶保存的不是当前while块的
                if(map_last_stack.get(list_identifier_use(i).code) != None){
                  //println("this_is_out_control(node)._2",this_is_out_control(node)._2,map_last_stack(list_identifier_use(i).code).head(0)._1)
                  if(this_is_out_control(node)._2 == map_last_stack(list_identifier_use(i).code).head(0)._1){
                    //连边，并将map_newuse_stack栈顶弹出
                    for(k <- 0 to map_last_stack(list_identifier_use(i).code).head.length-1){
                      for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
//                        builder.addEdge(map_last_stack(list_identifier_use(i).code).head(k)._2,map_newuse_stack(list_identifier_use(i).code).head._3(m),"LastUse")
                        builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_last_stack(list_identifier_use(i).code).head(k)._2,"LastUse")
                      }
                    }
                  }
                }
                else{
                  //这种情况可能都不存在
                  for(m <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                    if(map_newuse.get(list_identifier_use(i).code) == None){
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),list_identifier_use(i),"LastUse")
                    }else{
                      builder.addEdge(map_newuse_stack(list_identifier_use(i).code).head._3(m),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }
                  }
                }
                //将map_newuse_stack栈顶保存的当前while相关弹出
                map_newuse_stack(list_identifier_use(i).code).pop()
              }
            }

            if(! map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None){
//              println("out_do-while_map_newuse_stack",map_newuse_stack(list_identifier_use(i).code))
            }else{
//              println("out_do-while_map_newuse_stack,empty")
            }
            if(! map_last_stack.isEmpty && map_last_stack.get(list_identifier_use(i).code) != None){
//              println("out_do-while_map_last_stack",map_last_stack(list_identifier_use(i).code))
            }else{
//              println("out_do-while_map_last_stack,empty")
            }
          }
//          else if(this_is_out_control(node)._1 == "switch"){
//
//          }
        }
//        println("out_control_common_lastuse",map_last_stack(list_identifier_use(i).code))

      }


      //for循环开始为每个identifier连边
      //breakable{
      for (i <- 0 to list_identifier_use.length-1) {
        breakable {
          //提前结束当前一轮for循环的所有条件
          if (i == list_identifier_use.length - 1) {
            break
          }
          else if (next_same_name(i) == -1) { //如果找不到下一个同名的变量
            //to be continue 倒数第二个identifier还没有被处理
//            println("i没有被考虑", i)
            break
          }

          //当前变量与其下一同名变量的最大id公共父结点
          //如果当前变量没有下一同名变量
          var same_parent_maxid: AstNode = list_identifier_use(i)
          if (next_same_name(i) != -1) {
            same_parent_maxid = same_maxid_parent(list_identifier_use(i), list_identifier_use(next_same_name(i)))
//            println("same_parent_maxid.code",same_parent_maxid.code)
          }

          //获取当前节点和下一同名节点的类型，即是否为判断条件，是否为出结构体变量
          var this_is_judge: (String, AstNode) = is_judge(i)
          var next_is_judge: (String, AstNode) = is_judge(next_same_name(i))
          var this_is_out_control = is_out_control(i).toList.map { case (key, value) => (value, key) }
          var next_is_out_control = is_out_control(next_same_name(i)).toList.map { case (key, value) => (value, key) }
//          println(i, next_same_name(i), list_identifier_use(i).id, list_identifier_use(next_same_name(i)).id, list_identifier_use(i).code, list_identifier_use(next_same_name(i)).code, this_is_judge, this_is_out_control, next_is_judge, next_is_out_control)

          if(map_newuse_stack.get(list_identifier_use(i).code) != None){
//            println("map_newuse_stack_front",map_newuse_stack(list_identifier_use(i).code))
          }else{
//            println("map_newuse_stack_front","empty")
          }
          if(map_last_stack.get(list_identifier_use(i).code) != None){
//            println("map_last_stack_front",map_last_stack(list_identifier_use(i).code))
          }else{
//            println("map_last_stack_front","empty")
          }

          if (same_parent_maxid.code.contains("<empty>") || same_parent_maxid == list_identifier_use(i))
          {
            if(same_parent_maxid.astParent.isControlStructure && same_parent_maxid.astParent.code.contains("switch"))
            {
              //针对switch的不同分支之间的情况
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
//                println("switch-ynyy")
                //处理多层出块的情况
                out_control(i,this_is_out_control)
                //下一同名变量虽然为普通语句，但也有可能是入块（可能为多个）的变量，所以需要进行判断并通过判断后为块保存入块前数据流
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//                println("switch-ynyn")
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//                println("switch-ynny")
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
//                println("switch-ynnn")
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                //当前变量既是判断语句，又是出结构体，下一同名变量是else分支中的普通语句(需要分情况讨论)
//                println("switch-nnyy")
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//                println("switch-nnyn")
                out_control(i,this_is_out_control)
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//                println("switch-nnny")
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                //出判断语句进判断语句 ，当前变量既是判断语句的变量也是出结构体的变量
                //此种情况，肯定要将map_newuse_stack栈顶弹出栈，然后交还map_use（但这种情况，mapuse不会被更新），就直接使用连边，然后将下一同名变量入栈
//                println("switch-nnnn")
                out_control(i,this_is_out_control)
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
            }
            else
            {
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                //普通语句之间
//                println("yyyy")
                //common_iden(i)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //当前变量是普通变量，下一同名变量是出结构体的变量
                //有三种情况
                //1 排除 当前变量与下一同名变量不可能是不同结构体的变量，因为该情况下，当前变量必然是上一结构体的出变量，那么标识位不应该为false
                //2 当前变量如果是结构体外的同名变量，意味着下一同名变量的结构体的判读语句中没有该变量
                //3 当前变量与下一同名变量处于同一结构体内，还是需要考虑若是处于循环体内，要与控制体的判断语句进行交互
//                println("yyyn")
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
                //下一同名变量为出结构体变量时，且出的块为for时，且为for的内部变量时，需要走for_Inner的处理
                for_Inner(i,next_is_out_control(0)._1,next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
                //当前变量为普通语句，下一同名变量为结构体的判断语句
//                println("yyny")
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                //当前变量为普通变量，下一同名变量既是判断语句中又是出当前块的变量
//                println("yynn")
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if ((this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false")
                ||(this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false")) {
                //当前变量为出结构体进入普通语句（无论该普通变量是否在结构体内），就把当前栈顶的元素弹出去
                //当前情况属于出结构体有同名变量，但是也存在没有同名变量的情况
//                println("ynyy||ynyn")
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
                for_Inner(i,next_is_out_control(0)._1,next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
//                println("ynny")
                out_control_common(i,this_is_out_control)
                //将当前map_last_stack栈顶保存的出数据流，作为下一同名变量入块的入块前数据流
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)

                //                //跨不同结构体，当前变量为出块变量，下一同名变量为为判断语句变量
                //                //整理出结构体的数据流map_last_stack,连边之后，弹出栈, 并更新map_newuse数据流
                //                var list_out_if = new ListBuffer[AstNode]()
                //                for (node <- 0 to this_is_out_control.length-1){
                //                  breakable{
                //                    //首先判断出的块中的判断语句是否包含当前变量，也就是map_newuse_stack栈顶保存的控制节点与this_is_out_control(node)._2是否相同
                //                    //如果相同，则进行相关处理，如果不同，结束当层for循环，继续判断下一个
                //                    if (this_is_out_control(node)._1 == "while") {
                //                      if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2) {
                //                        //首先处理出块的判断变量的连边情况，连边后弹出栈顶
                //                        if (map_newuse.get(list_identifier_use(i).code) == None) {
                //                          for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                //                            builder.addEdge(list_identifier_use(i), map_newuse_stack(list_identifier_use(i).code).head._2(k), "LastUse")
                //                          }
                //                        } else {
                //                          //应该是因为进结构体的时候没有加入判断语句变量，
                //                          for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                //                            builder.addEdge(map_newuse(list_identifier_use(i).code).head, map_newuse_stack(list_identifier_use(i).code).head._2(k), "LastUse")
                //                          }
                //                        }
                //                        map_newuse_stack(list_identifier_use(i).code).pop()
                //                        update_map_newuse(i,list_identifier_use(next_same_name(i)))
                //                        //下一同名变量是入块的判断语句，因此需要入map_newuse_stack栈
                //                        map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                //                      }
                //                      else{
                //                        break()
                //                      }
                //                    }
                //                    else if (this_is_out_control(node)._1 == "if") {
                //                      if (order_or_nest(map_last_stack(list_identifier_use(i).code).head(0)._1, this_is_out_control(node)._2)) {
                //                        //下一同名变量是判断语句，直接连边就可以
                //                        builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), list_identifier_use(i), "LastUse")
                //
                //                        for (k <- 0 to map_last_stack(list_identifier_use(i).code).head.length - 1) {
                //                          builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), map_last_stack(list_identifier_use(i).code).head(k)._2, "LastUse")
                //                          list_out_if += map_last_stack(list_identifier_use(i).code).head(k)._2
                //                        }
                //                        //并且下一同名变量是判断语句，需要map_newuse_stack入栈，保留所有出if分支的数据流,并更新了数据流
                //                        if (map_newuse.get(list_identifier_use(i).code) == None) {
                //                          list_out_if += list_identifier_use(i)
                //                          map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(next_same_name(i)))
                //                        } else {
                //                          list_out_if += map_newuse(list_identifier_use(i).code).head
                //                          map_newuse(list_identifier_use(i).code).push(list_identifier_use(next_same_name(i)))
                //                        }
                //                        map_last_stack(list_identifier_use(i).code).pop()
                //                      }
                //                      else {
                //                        break()
                //                      }
                //                      //下一同名变量为入块的判断语句，无论是if还是while都需要入map_newuse_stack的栈
                //                      map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,list_out_if, ListBuffer(list_identifier_use(next_same_name(i)))))
                //                    }
                //                  }
                //                }
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
//                println("ynnn")
                out_control_common(i,this_is_out_control)
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)

                //                var list_out_if = new ListBuffer[AstNode]()
                //                for (node <- 0 to this_is_out_control.length-1){
                //                  breakable{
                //                    //首先判断出的块中的判断语句是否包含当前变量，也就是map_newuse_stack栈顶保存的控制节点与this_is_out_control(node)._2是否相同
                //                    //如果相同，则进行相关处理，如果不同，结束当层for循环，继续判断下一个
                //                    if (this_is_out_control(node)._1 == "while") {
                //                      if(map_newuse_stack(list_identifier_use(i).code).head._1 == this_is_out_control(node)._2) {
                //                        //首先处理出块的判断变量的连边情况，连边后弹出栈顶
                //                        if (map_newuse.get(list_identifier_use(i).code) == None) {
                //                          for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                //                            builder.addEdge(list_identifier_use(i), map_newuse_stack(list_identifier_use(i).code).head._2(k), "LastUse")
                //                          }
                //                        } else {
                //                          //应该是因为进结构体的时候没有加入判断语句变量，
                //                          for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                //                            builder.addEdge(map_newuse(list_identifier_use(i).code).head, map_newuse_stack(list_identifier_use(i).code).head._2(k), "LastUse")
                //                          }
                //                        }
                //                        map_newuse_stack(list_identifier_use(i).code).pop()
                //                        if (map_newuse.get(list_identifier_use(i).code) == None) {
                //                          map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(next_same_name(i)))
                //                        } else {
                //                          map_newuse(list_identifier_use(i).code).push(list_identifier_use(next_same_name(i)))
                //                        }
                //                        //下一同名变量是入块的判断语句,也是出块的语句，因此若入的块是while就不需要入map_newuse_stack栈，若入的块是if，说明入的是第一个分支，所以需要入栈
                //                        if(next_is_judge._1 == "if"){
                //                          map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,ListBuffer(list_identifier_use(i)), ListBuffer(list_identifier_use(next_same_name(i)))))
                //                        }
                //                      }else {
                //                        break()
                //                      }
                //                    }
                //                    else if (this_is_out_control(node)._1 == "if") {
                //                      if (order_or_nest(map_last_stack(list_identifier_use(i).code).head(0)._1, this_is_out_control(node)._2)) {
                //                        //下一同名变量是判断语句，直接连边就可以
                //                        builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), list_identifier_use(i), "LastUse")
                //
                //                        for (k <- 0 to map_last_stack(list_identifier_use(i).code).head.length - 1) {
                //                          builder.addEdge(list_identifier_use(next_same_name(next_same_name(i))), map_last_stack(list_identifier_use(i).code).head(k)._2, "LastUse")
                //                          list_out_if += map_last_stack(list_identifier_use(i).code).head(k)._2
                //                        }
                //                        //并且下一同名变量是判断语句，需要map_newuse_stack入栈，保留所有出if分支的数据流,并更新了数据流
                //                        if (map_newuse.get(list_identifier_use(i).code) == None) {
                //                          list_out_if += list_identifier_use(i)
                //                          map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(next_same_name(i)))
                //                        } else {
                //                          list_out_if += map_newuse(list_identifier_use(i).code).head
                //                          map_newuse(list_identifier_use(i).code).push(list_identifier_use(next_same_name(i)))
                //                        }
                //                        map_last_stack(list_identifier_use(i).code).pop()
                //                      }
                //                      else {
                //                        break()
                //                      }
                //                      //下一同名变量为入块的判断语句，无论是if还是while都需要入map_newuse_stack的栈
                //                      if(next_is_judge._1 == "if"){
                //                        map_newuse_stack(list_identifier_use(i).code).push((next_is_judge._2,list_out_if, ListBuffer(list_identifier_use(next_same_name(i)))))
                //                      }
                //                    }
                //                  }
                //                }
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                //此时公共父结点不是block.empty，
                // pass
//                println("nyyy no")
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //pass
//                println("nyyn no")
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false")  {
//                println("nyny no")
                //pass
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                //pass
//                println("nynn no")
              }

              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                //当前变量既是判断语句又是出结构体，下一同名变量是普通语句(需要分情况讨论)
//                println("nnyy")
                out_control_common(i, this_is_out_control)
                common_yy_yn(i, this_is_judge._1, this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //跨结构体：当前变量既是判断变量也是出控制体变量，下一同名变量是出下一出结构体变量（并且是非结构体内的普通变量）
                //为当前变量连边，因为是属于判断语句
//                println("nnyn")
                out_control_common(i, this_is_out_control)
                common_yy_yn(i,this_is_judge._1, this_is_out_control(0)._1)
                for_Inner(i,next_is_out_control(0)._1, next_is_out_control(0)._2)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//                println("nnny")
                out_control_common(i, this_is_out_control)
                common_ny_nn(i, this_is_judge._1, this_is_out_control(0)._1)

              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
//                println("nnnn")
                out_control_common(i,this_is_out_control)
                common_ny_nn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
            }
          }
          else if (same_parent_maxid.code.contains("while"))
          {
            //判断条件（变量）进循环体（同名变量）
            //但是进循环体有可能是普通语句，也有可能是又一个循环体的判断条件或普通语句或者出结构体语句
            //while(x>0){if(x>5){...}}
            if(same_parent_maxid.code.contains("do")){
              //do-while
              if (this_is_judge._1 == "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //普通变量进出结构体语句
//                println("do-while-yyyn")
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //判断体进结构体内普通语句，并且该同名变量为出变量
//                println("do-while-ynyn")
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //当前变量即为判断语句变量又为出结构体语句，下一同名变量为出do-while块变量，且该变量在while判断语句里
//                println("do-while-nnyn")
                out_control_common(i,this_is_out_control)
                common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
              }
            }
            else{ //while
              if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
                //判断体进结构体内普通语句
                //将栈顶元素与同名变量连接，并更新map_use
//                println("while-nyyy")
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
                //判断体进结构体内普通语句，并且该同名变量为出变量
//                println("while-nyyn")
                if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false") {
                //判断体进结构体内普通语句，并且该同名变量为出变量
//                println("while-nyny")
                if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
              }
              else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
                //判断体进结构体内普通语句，并且该同名变量为出变量
//                println("while-nynn")
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
            //普通变量是1，2，3三个分支，正常连数据流边，出结构体也在前三个分支的情况，需要保存然后连回前三个分支
            //for内部限定变量，
            //如何区分两个变量，直接找当前变量的上一个同名变量，如果找不到就用一个数据结构保存一下？，因为for内部的限定变量是不会出现在for外部的
//            println("test",list_identifier_use(i),for_Belong_Which(list_identifier_use(i),same_parent_maxid))
            if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//              println("for-nyny")
              var for_Belong_This: Int = for_Belong_Which(list_identifier_use(i),this_is_judge._2)
              var for_Belong_Next: Int = for_Belong_Which(list_identifier_use(next_same_name(i)),next_is_judge._2)
              //记录独属于for判断体作用域的内部变量
              if(map_For_Inner.get(same_parent_maxid) != None && !map_For_Inner(same_parent_maxid).isEmpty)
              {
                //如果当前不为空，再进行下一步的判断
                if(!map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code)){
                  //如果当前变量不存在这里面
                  if(last_same_name(i) == -1){
                    //如果当前变量不存在上一同名变量说明是内部变量，
                    map_For_Inner(same_parent_maxid) += list_identifier_use(i).code
                    //万一外部变量也不存在怎么办呢(大概率是不存在的吧)
                    //第二个分支的变量就相当于while里的判断语句变量
                  }
                }
              }
              else
              {//如果为空
//                println("last_same_name(i)",last_same_name(i))
                if(last_same_name(i) == -1){
                  map_For_Inner(same_parent_maxid) = ListBuffer(list_identifier_use(i).code)
                  //map_For_Inner(same_parent_maxid) += list_identifier_use(i).code
                }
              }
//              println("map_For_Inner",map_For_Inner)

              //先根据下一同名变量是否是第四个分支也就是for块来进行区分每种情况再根据当前变量为外部变量还是内部变量分情况讨论
              if(for_Belong_Next == 4){
                if(for_Belong_This == 1){

                }
                else if(for_Belong_This == 2){

                }else if(for_Belong_This == 3){

                }
              }else{
                if(for_Belong_This == 1 && for_Belong_Next == 2)
                {
//                  println("same_parent_maxid",same_parent_maxid)
                  if(! map_For_Inner.isEmpty && map_For_Inner.get(same_parent_maxid) != None && ! map_For_Inner(same_parent_maxid).isEmpty){
                    if(map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code))
                    {
//                      println("inner 1 2")
                      if(map_For_Record.get(same_parent_maxid) == None){
                        map_For_Record(same_parent_maxid) = ListBuffer(true,true)
                      }else{
                        map_For_Record(same_parent_maxid)(0) = true
                        map_For_Record(same_parent_maxid)(1) = true
                      }
                      val hashMap2 = new mutable.HashMap[String, ListBuffer[AstNode]]
                      hashMap2(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                      map_For_Second(same_parent_maxid) = hashMap2
//                      println("map_For_Second",map_For_Second)
                      //                    map_For_Second(same_parent_maxid)(list_identifier_use(i).code) = ListBuffer(list_identifier_use(next_same_name(i)))
                      //内部变量
                      builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(i),"LastUse")
                      //为内部变量保存块的判断语句变量，由于没有入块前数据流，所以用第一分支的变量暂时填充了这个位置，目前是没有实质意义的
                      if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                        map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }else{
                        map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }
                    }
                  }
                  else
                  {
//                    println("outer 1 2")
                    // 外部变量
                    //有第一个分支，说明之前入块时已经把第一个分支
                    // 入for循环时，map_newuse不要更新，保留入for前的数据流, 直接连边就可以
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    // 为第二分支的判断语句的外部变量保存
                    if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                      map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }else{
                      map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }
                  }
                }
                else if(for_Belong_This == 1 && for_Belong_Next == 3)
                {
                  //说明直接没有第二个分支的存在，伪代码很多都是这种情况,也需要为第一个分支保存变量
                  if(! map_For_Inner.isEmpty && map_For_Inner.get(same_parent_maxid) != None && ! map_For_Inner(same_parent_maxid).isEmpty){
                    if(map_For_Inner(same_parent_maxid).contains(list_identifier_use(i).code))
                    {
//                      println("inner 1 3")
                      if(map_For_Record.get(same_parent_maxid) == None){
                        map_For_Record(same_parent_maxid) = ListBuffer(true,false,true)
                      }else{
                        map_For_Record(same_parent_maxid)(0) = true
                        map_For_Record(same_parent_maxid)(1) = false
                        map_For_Record(same_parent_maxid)(2) = true
                      }

                      //为第一个分支保存变量
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
                      //为第三个分支保存变量(记录为了之后出for时连边使用)
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
                        //为i++ i-- ++i --i这些情况以及变种连边（自己连自己）
                        if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                          builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                        }
                      }

                      //更新数据流,因为第三分支的数据等块执行完再连边，因此，如果没有第二个分支的情况，保存第一个分支的数据流是为了与块的（入块的第一个普通变量）同名变量相连
                      if(map_newuse.get(list_identifier_use(i).code) != None){
                        map_newuse(list_identifier_use(i).code).push(list_identifier_use(i))
                      }else{
                        map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(i))
                      }
                    }
                  }
                  else
                  {//如果是外部变量
//                    println("outer 1 3")
                    //入for循环时，map_newuse不要更新，保留入for前的数据流,直接连边就可以
                    if(! map_newuse.isEmpty && map_newuse.get(list_identifier_use(i).code) != None && ! map_newuse(list_identifier_use(i).code).isEmpty){
                      builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                    }

                    //记录为了之后出for时连边使用
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
//                      println("inner 2 3")
                      if(map_For_Record.get(same_parent_maxid) == None){
                        //说明之前没有记录1、2，说明第一个分支不存在
                        map_For_Record(same_parent_maxid) = ListBuffer(false,true,true)
                      }else{
                        //说明1、2分支之前已经记录，只需要记录一下3就可以
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
                        //为i++ i-- ++i --i这些情况以及变种连边（自己连自己）
                        if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                          builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                        }
                      }

                      if(map_For_Record.get(same_parent_maxid) != None){
                        if(map_For_Record(same_parent_maxid)(0) == false){
                          //如果第一分支不存在，在此处将第二分支保存的条件变量入栈
                          //相当于while的判断语句
                          if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                            map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                          }else{
                            map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(list_identifier_use(i)),ListBuffer(list_identifier_use(next_same_name(i)))))
                          }
                        }
                        //为第二分支与第三分支的同名变量连边
                        if(map_newuse_stack.get(list_identifier_use(i).code) != None){
                          for(k2 <- 0 to map_newuse_stack(list_identifier_use(i).code).head._3.length-1){
                            builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse_stack(list_identifier_use(i).code).head._3(k2),"LastUse")
                          }
                        }
                      }

                      //                    //更新数据流,因为第三分支的数据等块执行完再连边，因此，如果没有第二个分支的情况，保存第一个分支的数据流是为了与块的同名变量相连
                      //                    if(map_newuse.get(list_identifier_use(i).code) != None){
                      //                      map_newuse(list_identifier_use(i).code).push(list_identifier_use(i))
                      //                    }else{
                      //                      map_newuse(list_identifier_use(i).code) = Stack(list_identifier_use(i))
                      //                    }
                    }
                  }
                  else
                  {//外部变量
//                    println("outer 2 3")
                    //根据第一分支是否存在同名变量，决定是否添加for的当前变量的map_newuse_stack,并且为第二个分支的变量连边
                    if(map_newuse_stack.get(list_identifier_use(i).code) == None){
                      //不存在第一分支
                      builder.addEdge(list_identifier_use(i),map_newuse(list_identifier_use(i).code).head,"LastUse")
                      map_newuse_stack(list_identifier_use(i).code) = Stack((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                    }else{
                      if(map_newuse_stack(list_identifier_use(i).code).head._1 != same_parent_maxid){
                        //不存在第一分支
                        builder.addEdge(list_identifier_use(i),map_newuse(list_identifier_use(i).code).head,"LastUse")
                        map_newuse_stack(list_identifier_use(i).code).push((same_parent_maxid,ListBuffer(map_newuse(list_identifier_use(i).code).head),ListBuffer(list_identifier_use(next_same_name(i)))))
                      }
                    }

                    //为第三分支的同名变量与入块前数据流连边
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")

                    for(k <- 0 to list_In_Decrement.length-1){
                      //为i++ i-- ++i --i这些情况以及变种连边（自己连自己）
                      if(list_identifier_use(next_same_name(i)).astParent.code.contains(list_In_Decrement(k))){
                        builder.addEdge(list_identifier_use(next_same_name(i)),list_identifier_use(next_same_name(i)),"LastUse")
                      }
                    }
                  }
                }
              }

              //if_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)

            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
//              println("for-nyyy")
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//              println("for-nyyn")
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
//              println("for-nynn")
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
          }
          else if (same_parent_maxid.code.contains("if"))
          {//共12种情况，没有yy__的四种情况
            if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
//              println("if-ynyy")
              //处理多层出块的情况
              out_control(i,this_is_out_control)
              //下一同名变量虽然为普通语句，但也有可能是入块（可能为多个）的变量，所以需要进行判断并通过判断后为块保存入块前数据流
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//              println("if-ynyn")
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//              println("if-ynny")
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 == "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
//              println("if-ynnn")
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false"){
//              println("if-nyyy")
              //下一同名变量虽然为普通语句，但是也有可能是入块的变量
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//              println("if-nyyn")
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//              println("if-nyny")
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false"){
//              println("if-nynn")
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
              //当前变量既是判断语句，又是出结构体，下一同名变量是else分支中的普通语句(需要分情况讨论)
//              println("if-nnyy")
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false"){
//              println("if-nnyn")
              out_control(i,this_is_out_control)
              if_yy_yn(i,this_is_judge._1,this_is_judge._2,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 == "false"){
//              println("if-nnny")
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 != "false" && next_is_judge._1 != "false" && next_is_out_control(0)._1 != "false") {
              //出判断语句进判断语句 ，当前变量既是判断语句的变量也是出结构体的变量
              //此种情况，肯定要将map_newuse_stack栈顶弹出栈，然后交还map_use（但这种情况，mapuse不会被更新），就直接使用连边，然后将下一同名变量入栈
//              println("if-nnnn")
              out_control(i,this_is_out_control)
              if_ny_nn(i,this_is_judge,next_is_judge,this_is_out_control)
            }
          }
          else if (same_parent_maxid.code.contains("switch"))
          {
//            println("switch",i,list_identifier_use(i))
            if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 == "false") {
              //判断语句入分支变量,虽然入switch的case分支可能是入块，但是switch是靠find_between_control来判定是入case和入default分支的
//              println("switch-nyyy")
              common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
            }
            else if (this_is_judge._1 != "false" && this_is_out_control(0)._1 == "false" && next_is_judge._1 == "false" && next_is_out_control(0)._1 != "false") {
              //当前变量为判断语句，下一同名变量同时为入分支变量和出分支变量，但是下一同名变量的标志是yn是因为switch的入分支是在find_between_control体现的
//              println("switch-nyyn")
              common_yy_yn(i,this_is_judge._1,this_is_out_control(0)._1)
            }
          }
          else if (same_parent_maxid.code.contains("="))
          {
            if (same_parent_maxid.code.contains("==")) {
              //前后两个变量都在判断语句里，eg:x == set(x,y)
              //将这两个变量都加入map_new_stack栈顶里
//              map_newuse_stack(list_identifier_use(i).code).head._3 += list_identifier_use(next_same_name(i))
//              //连接map_newuse_stack栈顶保存的当前块的入块前数据流，由于是listbuffer结构，使用for循环连接
//              for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
//                builder.addEdge(map_newuse(list_identifier_use(i).code).head, map_newuse_stack(list_identifier_use(i).code).head._2(k) ,"LastUse")
//              }

              if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
                map_newuse_stack(list_identifier_use(i).code).head._3 += list_identifier_use(next_same_name(i))
                //连接map_newuse_stack栈顶保存的当前块的入块前数据流，由于是listbuffer结构，使用for循环连接
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
//            println("&&")
            //主要为了保存判断语句的变量，方便出块的时候使用(while和for)
            //存在当前变量在map_newuse_stack没有对应的元素的情况，比如for的第一个分支没有，那么就不会进入if判断（当无处理）
            if(!map_newuse_stack.isEmpty && map_newuse_stack.get(list_identifier_use(i).code) != None && ! map_newuse_stack(list_identifier_use(i).code).isEmpty){
              map_newuse_stack(list_identifier_use(i).code).head._3 += list_identifier_use(next_same_name(i))
              //连接map_newuse_stack栈顶保存的当前块的入块前数据流，由于是listbuffer结构，使用for循环连接
              for (k <- 0 to map_newuse_stack(list_identifier_use(i).code).head._2.length - 1) {
                builder.addEdge(list_identifier_use(next_same_name(i)), map_newuse_stack(list_identifier_use(i).code).head._2(k) ,"LastUse")
              }
            }
            //对于for结构比较特殊，第二分支的判断变量需要包含在map_For_Second里
            if(map_newuse_stack.get(list_identifier_use(i).code) != None){
              var for_node = map_newuse_stack(list_identifier_use(i).code).head._1
              if(map_For_Record.get(for_node) != None){
                if(map_For_Record(for_node)(1) == true){//如果第一分支存在
                  map_For_Second(for_node)(list_identifier_use(i).code) += list_identifier_use(next_same_name(i))
                  if(!map_For_Inner(for_node).contains(list_identifier_use(i).code)){
                    //如果不是内部特有的变量
                    builder.addEdge(list_identifier_use(next_same_name(i)),map_newuse(list_identifier_use(i).code).head,"LastUse")
                  }
                }
              }
            }
            //并且通过map_newuse连边,因为在此变量之前的判断语句里也有该变量，当时没有更新数据流
            //builder.addEdge(list_identifier_use(next_same_name(i)), map_newuse(list_identifier_use(i).code).head ,"LastUse")
          }
          if(map_newuse_stack.get(list_identifier_use(i).code) != None){
//            println("map_newuse_stack_back",map_newuse_stack(list_identifier_use(i).code))
          }else{
//            println("map_newuse_stack_back","empty")
          }
          if(map_last_stack.get(list_identifier_use(i).code) != None){
//            println("map_last_stack_back",map_last_stack(list_identifier_use(i).code))
          }else{
//            println("map_last_stack_back","empty")
          }
        }
      }
      //}
    })
  }

  override def run(builder: BatchedUpdate.DiffGraphBuilder): Unit = {
    add_computedFrom(builder)
//    add_lastwrite(builder)
    add_lastuse(builder)

    //unable to reload function:addEdge because of the wrong type of parameters
    //    builder.addEdge(cpg.identifier.name("x"),cpg.identifier.name("y"),"ast")

  }
}
