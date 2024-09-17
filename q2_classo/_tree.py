import skbio
import numpy as np

from plotly import graph_objects

colors = {"threshold": "#8e0202", "selected": "#f74e6b", "unselected": "#68025e"}


def make_lists_tree(sktree, circular=False):

    POS, E, LAB = [], [], []

    def add(tet, r, node, dtet: float, circular=False):
        current_index = len(POS)
        LAB.append(node.name)
        CHILDREN = node.children
        n = len(CHILDREN)
        if n > 0:

            if circular:
                POS.append(from_polaire(r, tet))
            else:
                POS.append([tet, r])
            step = dtet / n
            tet_new = tet - dtet - step
            depth = 0
            for i in range(n):
                tet_new = tet_new + 2 * step
                E.append([current_index, len(POS)])
                child_depth = add(
                    tet_new, r + 1, CHILDREN[i], step, circular=circular
                )
                depth = max(depth, child_depth)
            return depth + 1

        else:
            if circular:
                POS.append(from_polaire(r, tet))
            else:
                if node.name[1] == "_":
                    POS.append([tet, r])
                else:
                    POS.append([tet, "depth"])
            return 0

    depth = add(0.0, 0.0, sktree, np.pi, circular=circular)
    for i in range(len(LAB)):
        if LAB[i] is None:
            LAB[i] = "None"
        # if LAB[i][-1] == '_' : remove(POS,E,LAB,i)
        # if not LAB[i] in labels : remove(POS,E,LAB,i)

    for c in POS:
        if type(c[1]) == str:
            c[1] = depth

    return -np.array(POS), E, np.array(LAB)


def remove(POS, E, LAB, i):
    LAB[i] = "None"
    POS[i] = [0.0, 0.0]
    pred = 0
    for k, c in enumerate(E):
        if c[1] == i:
            pred = c[0]
            E.pop(k)
            break
    for c in E:
        if c[0] == i:
            c[0] = pred


def from_polaire(r, tet):
    x = r * np.cos(tet)
    y = r * np.sin(tet)
    return [x, y]


def tree_to_matrix(tree, label, with_repr=False):
    # to do here : given a skbio tree and the beta-labels in a given order,
    # return the matrix A and the new labels corresponding
    dicti = dict()
    d = len(label)
    LEAVES = [tip.name for tip in tree.tips()]
    order = []
    # list that will give the order in which the nodes are added
    # in the dicti, such that it will be easy to remove similar nodes
    for i, name_leaf in enumerate(label):
        name_leaf = label[i]
        dicti[name_leaf] = np.zeros(d, dtype=bool)
        dicti[name_leaf][i] = True
        order.append(name_leaf)
        if name_leaf not in LEAVES:
            tree.append(
                skbio.TreeNode(name=name_leaf)
            )  # add the node if it is node already in the tree
            print(
                "The feature {} i not in the leaves of the tree".format(
                    name_leaf
                )
            )
        for n in tree.find(name_leaf).ancestors():
            ancest = n.name
            if ancest[-1] != "_":
                if ancest not in dicti:
                    dicti[ancest] = np.zeros(d, dtype=bool)
                    order.append(ancest)
                dicti[ancest][i] = True

    L, label2 = [], []

    for node in tree.levelorder():
        nam = node.name
        if nam in dicti and nam not in label2:
            label2.append(nam)
            L.append(dicti[nam])

    to_keep = np.ones(len(L), dtype=bool)
    to_keep = remove_same_vect(L, label2, order)
    L = np.array(L)
    to_keep[np.all(L, axis=1)] = False

    return L[to_keep].T, np.array(label2)[to_keep]


def remove_same_vect(L, label, order):
    K = len(L)
    to_keep = np.array([True] * K)
    j = label.index(order[-1])
    col = L[j]
    for i in range(K - 2, -1, -1):
        new_j = label.index(order[i])
        new_col = L[new_j]
        if np.array_equal(col, new_col):
            to_keep[new_j] = False
        else:
            j, col = new_j, new_col

    return to_keep


def make_annotations(pos, lab, font_size=10, font_color="rgb(250,250,250)"):
    annotations = []
    for k in range(len(pos)):
        annotations.append(
            dict(
                text=lab[k],  # text within the circles
                x=pos[k, 0],
                y=pos[k, 1],
                xref="x1",
                yref="y1",
                font=dict(color=font_color, size=font_size),
                showarrow=False,
            )
        )
    return annotations


def plot_tree(taxa, labels, selected_labels=None, circular=False):

    position, Edges, labels_nodes = make_lists_tree(
        build_subtree(taxa, labels), circular=circular
    )

    # position[:,1] = 2*max(position[:,1]) - position[:,1]

    Xe, Ye = [], []
    for edge in Edges:
        Xe += [position[edge[0], 0], position[edge[1], 0], None]
        Ye += [position[edge[0], 1], position[edge[1], 1], None]

    selected = np.array([False] * len(position))

    if selected_labels is not None:
        for i in range(len(position)):
            if labels_nodes[i] in selected_labels:
                selected[i] = True

    unselected = np.array([not i for i in selected])

    fig = graph_objects.Figure()

    # plot the edges
    fig.add_trace(
        graph_objects.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            line=dict(color="rgb(210,210,210)", width=1),
            hoverinfo="none",
        )
    )

    # plot the nodes not selected
    fig.add_trace(
        graph_objects.Scatter(
            x=position[unselected, 0],
            y=position[unselected, 1],
            mode="markers",
            name="nodes",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color=colors["unselected"],
                line=dict(color="black", width=1),
            ),
            text=labels_nodes[unselected],
            hoverinfo="text",
            opacity=0.8,
        )
    )

    # plot the nodes selected
    fig.add_trace(
        graph_objects.Scatter(
            x=position[selected, 0],
            y=position[selected, 1],
            mode="markers",
            name="nodes",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color=colors["selected"],
                line=dict(color="black", width=1),
            ),
            text=labels_nodes[selected],
            hoverinfo="text",
            opacity=0.8,
        )
    )

    axis = dict(
        showline=True,  # hide axis line, grid, ticklabels and  title
        zeroline=True,
        showgrid=True,
        showticklabels=True,
    )

    annot = []
    for name in labels_nodes:
        if name[1] == "_":
            annot.append(name[0])
        else:
            annot.append("B")

    fig.update_layout(
        title="Taxonomic tree",
        annotations=make_annotations(position, annot),
        font_size=12,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
    )

    return fig


def remove2(tree, node):
    if node.parent is not None:
        for child in node.children:
            child.parent = node.parent
            node.parent.children.append(child)

        tree.remove_deleted(lambda x: x.name == node.name)


def build_subtree(sktree, label_node):
    sub_sktree = sktree.copy()
    for node in sub_sktree.preorder():
        if node.name not in label_node:
            remove2(sub_sktree, node)
    return sub_sktree
