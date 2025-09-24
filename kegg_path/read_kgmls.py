import xml.etree.ElementTree as ET
import os


kgml_files = os.listdir("./kgmls")

kgml_files = [f for f in kgml_files if ".sh" not in f]
bad_trees = []

for file in kgml_files:
    graph_name = file.split(".")[0][3:]
    print(graph_name)
    try:
        tree = ET.parse("kgmls/hsa{}.kgml".format(graph_name))
    except(ET.ParseError):
        bad_trees.append(graph_name)
        continue
    root = tree.getroot()


    os.makedirs("./processed/{}".format(graph_name), exist_ok=True)


    # --- Nodes ---
    nodes = []
    for e in root.findall("entry"):
        node_id = e.attrib["id"]
        ntype   = e.attrib.get("type","")
        graphics = e.find("graphics")
        name = graphics.attrib.get("name")

        if(name == None):
                name    = e.attrib.get("name","")  # space-separated IDs (e.g., hsa:######## or ko:K####)

        nodes.append((node_id, ntype, name))

    with open("processed/{}//nodes.tsv".format(graph_name),"w") as f:
        f.write("id\ttype\tname\n")
        for r in nodes:
            f.write("\t".join(r) + "\n")

    # --- Edges from <relation> ---
    edges = []
    for r in root.findall("relation"):
        src = r.attrib["entry1"]
        dst = r.attrib["entry2"]
        rtype = r.attrib.get("type","")       # e.g., PPrel, ECrel, GErel, etc.
        # subtypes (e.g., activation, inhibition)
        subs = [st.attrib.get("name","") for st in r.findall("subtype")]
        edges.append((src, dst, rtype, "|".join(subs)))

    with open("processed/{}/edges.tsv".format(graph_name),"w") as f:
        f.write("source\ttarget\trelation_type\tsubtypes\n")
        for e in edges:
            f.write("\t".join(e) + "\n")

    # --- (Optional) Edges from <reaction> connecting compounds/enzymes ---
    rxn_edges = []
    for rxn in root.findall("reaction"):
        rid = rxn.attrib.get("id","")
        rtype = rxn.attrib.get("type","")     # e.g., reversible/irreversible
        # substrates and products:
        subs  = [s.attrib["name"] for s in rxn.findall("substrate")]
        prods = [p.attrib["name"] for p in rxn.findall("product")]
        for s in subs:
            for p in prods:
                rxn_edges.append((s, p, f"reaction:{rtype}", rid))
    with open("processed/{}/_reaction_edges.tsv".format(graph_name),"w") as f:
        f.write("substrate\tproduct\treaction_type\treaction_id\n")
        for e in rxn_edges:
            f.write("\t".join(e) + "\n")

print(bad_trees)