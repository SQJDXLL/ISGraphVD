import os 
import pydot
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from argparse import ArgumentParser

parser = ArgumentParser("according diff dot draw hightlight graph")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", action="store_true", default=False)
args = parser.parse_args()

def add_highlight(graphFile, diffGraphFile, outputHighlightGraph):
    graph = pydot.graph_from_dot_file(graphFile)[0]
    diffGraph = pydot.graph_from_dot_file(diffGraphFile)[0]

    graphNodes = graph.get_nodes()
    diffGraphNodes = diffGraph.get_nodes()
    graphNodeNameSet = set(map(lambda x:x.get_name().strip('"'), graphNodes))
    diffGraphNodeSet = set(map(lambda x:x.get_name(), diffGraphNodes))
    # print(graphNodeNameSet, diffGraphNodeSet)

    graphEdges = graph.get_edges()
    diffGraphEdges = diffGraph.get_edges()
    graphEdgeNames = set(map(lambda x: (x.get_source().strip('"'), x.get_destination().strip('"')), graphEdges))
    diffGraphEdgeNames = set(map(lambda x: (x.get_source(), x.get_destination()), diffGraphEdges))
    # print("graphEdgeNames, diffGraphEdgeNames", graphEdgeNames, diffGraphEdgeNames)

    NodeHighlight = [ ]
    for graphNode in graphNodeNameSet:
        for diffGraphNode in diffGraphNodeSet:
            if graphNode == diffGraphNode:
                NodeHighlight.append(graphNode)
    # print("NodeHighlight")

    EdgeHighlight = []
    for graphEdge in graphEdgeNames:
        for diffGraphEdge in diffGraphEdgeNames:
            if graphEdge == diffGraphEdge:
                EdgeHighlight.append(graphEdge)
    # print("EdgeHighlight", EdgeHighlight)

    for node in graphNodes:
        node_name = node.get_name().strip('"') 
        if node_name in NodeHighlight:
            # print('red')
            node.set_fillcolor("red")
        else:
            node.set_fillcolor("green")
            # print('green')
    for edge in graphEdges:
        edge_source = edge.get_source().strip('"')
        edge_destination = edge.get_destination().strip('"')
        # print("(edge_source, edge_destination)", (edge_source, edge_destination))
        if (edge_source, edge_destination) in EdgeHighlight:
            edge.set_color("red")
        else:
            edge.set_color("green")
    
    # 将更新后的图写入新的 DOT 文件
    dot_string = graph.to_string()

    # 写入高亮的 DOT 文件
    output_dot_file = os.path.join(outputHighlightGraph, "ast_deform.dot")
    with open(output_dot_file, "w") as f:
        f.write(dot_string)

def runner(task: tuple):
    graphFile, diffGraphFile, outputHighlightGraph = task
    # print(store_path)
    return add_highlight(graphFile, diffGraphFile, outputHighlightGraph)



if __name__ == "__main__":

    proj, cve, rl = args.project, args.cve_id, args.RL
    if rl:
        graph_path = os.path.join("../../realWorld/data/", proj, cve, "graph")
        diff_graph_path = os.path.join("../../realWorld/data/", proj, cve, "diffDot_hl")
        outputDir = os.path.join("../../realWorld/data/", proj, cve, "graph_hl")
    else:
        graph_path = os.path.join("../../data/", proj, cve, "graph")
        diff_graph_path = os.path.join("../../data/", proj, cve, "diffDot_hl")
        outputDir = os.path.join("../../data/", proj, cve, "graph_hl")

    tasks = []
    for diffDot in os.listdir(diff_graph_path):
        graphFile = os.path.join(graph_path, diffDot, "ast_deform.dot")
        diffGraphFile = os.path.join(diff_graph_path, diffDot, "ast_deform.dot")
        outputHighlightGraph = os.path.join(outputDir, diffDot)
        if not os.path.exists(outputHighlightGraph):
            os.makedirs(outputHighlightGraph)
        tasks.append((graphFile, diffGraphFile, outputHighlightGraph))
        
    res = process_map(runner, tasks, max_workers=10, chunksize=4)
        

        
        


