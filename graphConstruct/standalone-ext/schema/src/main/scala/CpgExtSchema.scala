import io.shiftleft.codepropertygraph.schema._
import overflowdb.schema.SchemaBuilder
import overflowdb.schema.Property.ValueType

class CpgExtSchema(builder: SchemaBuilder, cpgSchema: CpgSchema) {

  // Add node types, edge types, and properties here

  val myProperty = builder
    .addProperty(name = "MYPROPERTY", valueType = ValueType.String)
    .mandatory("")

  val myNodeType = builder
    .addNodeType("MYNODETYPE")
    .addProperty(myProperty)

  val ComputedFrom = builder
    .addEdgeType(name = "ComputedFrom",comment = "new edge for different variables")

  val LastUse = builder
    .addEdgeType(name = "LastUse",comment = "new edge for different variables")

  val LastWrite = builder
    .addEdgeType(name = "LastWrite",comment = "new edge for different variables")


//  cpgSchema.ast.identifier.addOutEdge(edge = cpgSchema.ast.ast,inNode = cpgSchema.ast.identifier)
  cpgSchema.ast.identifier.addOutEdge(edge = ComputedFrom,inNode = cpgSchema.ast.identifier)
  cpgSchema.ast.identifier.addOutEdge(edge = LastUse,inNode = cpgSchema.ast.identifier)
  cpgSchema.ast.identifier.addOutEdge(edge = LastWrite,inNode = cpgSchema.ast.identifier)
}

object CpgExtSchema {
  val builder   = new SchemaBuilder(domainShortName = "Cpg", basePackage = "io.shiftleft.codepropertygraph.generated")
  val cpgSchema = new CpgSchema(builder)
  val cpgExtSchema = new CpgExtSchema(builder, cpgSchema)
  val instance     = builder.build
}
