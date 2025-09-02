import os 
import pydot
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from argparse import ArgumentParser

parser = ArgumentParser("according diff dot draw hightlight graph")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", action="store_true", default=False)
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
args = parser.parse_args()

def add_highlight(graphFile, diffGraphFile, outputHighlightGraph):
    graph = pydot.graph_from_dot_file(graphFile)[0]
    diffGraph = pydot.graph_from_dot_file(diffGraphFile)[0]

    graphNodes = graph.get_nodes()
    diffGraphNodes = diffGraph.get_nodes()
    graphNodeNameSet = set(map(lambda x:x.get_name().strip('"'), graphNodes))
    diffGraphNodeSet = set(map(lambda x:x.get_name(), diffGraphNodes))

    graphEdges = graph.get_edges()
    diffGraphEdges = diffGraph.get_edges()
    graphEdgeNames = set(map(lambda x: (x.get_source().strip('"'), x.get_destination().strip('"')), graphEdges))
    diffGraphEdgeNames = set(map(lambda x: (x.get_source(), x.get_destination()), diffGraphEdges))

    NodeHighlight = [ ]
    for graphNode in graphNodeNameSet:
        for diffGraphNode in diffGraphNodeSet:
            if graphNode == diffGraphNode:
                NodeHighlight.append(graphNode)

    EdgeHighlight = []
    for graphEdge in graphEdgeNames:
        for diffGraphEdge in diffGraphEdgeNames:
            if graphEdge == diffGraphEdge:
                EdgeHighlight.append(graphEdge)

    for node in graphNodes:
        node_name = node.get_name().strip('"') 
        if node_name in NodeHighlight:
            node.set_fillcolor("red")
        else:
            node.set_fillcolor("green")
    for edge in graphEdges:
        edge_source = edge.get_source().strip('"')
        edge_destination = edge.get_destination().strip('"')
        if (edge_source, edge_destination) in EdgeHighlight:
            edge.set_color("red")
        else:
            edge.set_color("green")
    
    dot_string = graph.to_string()

    output_dot_file = os.path.join(outputHighlightGraph, "ast_deform.dot")
    with open(output_dot_file, "w") as f:
        f.write(dot_string)

def runner(task: tuple):
    graphFile, diffGraphFile, outputHighlightGraph = task
    return add_highlight(graphFile, diffGraphFile, outputHighlightGraph)



if __name__ == "__main__":

    if args.mission_id:
        proj, cve, rl, mission_id  = args.project, args.cve_id, args.RL, args.mission_id
    else:
        proj, cve, rl  = args.project, args.cve_id, args.RL
    if rl:
        graph_path = os.path.join("../../data_detect/data/", mission_id, proj, cve, "graph")
        diff_graph_path = os.path.join("../../data_detect/data/", mission_id, proj, cve, "diffDot_hl")
        outputDir = os.path.join("../../data_detect/data/", mission_id, proj, cve, "graph_hl")
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
        

        
        


