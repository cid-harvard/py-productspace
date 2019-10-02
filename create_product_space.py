import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import count
import json
import networkx as nx


def create_product_space(df_plot_dataframe=None,
                         df_plot_node_col=['node'],
                         df_plot_attribute_cols=['color', 'node_size']):
    """
    Creates customizable product space visualizations similar to the Atlas

    Args:
        df_plot_dataframe: DataFrame with node attributes
        df_plot_node_col: Column corresponding to the node name
        df_plot_attribute_cols: Columns corresponding to the noe attributes

    Returns:
        Plot object containing the product space visualization
    """
    # Load coordinates of nodes (original Atlas file)
    networkjs = json.load(open('data/network_hs92_4digit.json'))

    # Get nodes +  positions from json format into python list + dictionary
    nodes = []
    nodes_pos = {}
    for x in networkjs['nodes']:
        nodes.append(int(x['id']))
        nodes_pos[int(x['id'])] = (int(x['x']), -int(x['y'] / 1.5))

    # Get product space edge list (based on strength from the json)
    edges = []
    for x in networkjs['edges']:
        if x['strength'] > 1 or 1 == 1:
            edges.append((int(x['source']), int(x['target'])))

    # Create pandas dataframe from edges file
    dfe = pd.DataFrame(edges)
    dfe.rename(columns={0: 'src'}, inplace=True)
    dfe.rename(columns={1: 'trg'}, inplace=True)

    # from the data to be plotted, only select edges of nodes that are also present in product space
    dfe2 = pd.DataFrame(np.append(dfe['src'].values, dfe['trg'].values))
    dfe2.drop_duplicates(inplace=True)
    dfe2.rename(columns={0: 'node'}, inplace=True)
    dfn2 = pd.merge(df_plot_dataframe, dfe2, how='inner',
                    left_on=df_plot_node_col, right_on='node')

    # Now create networkx objects:
    # - G = only products from trade data that will be plotted
    G = nx.from_pandas_edgelist(dfn2, 'hs_product_code', 'hs_product_code')

    # - G2 = all nodes + edges from original product space
    # -- these will be gray in the background, i.e. products for which there is no info
    # -- in this example, only products from Ukraine with exports > $ 40 mil.
    G2 = nx.from_pandas_edgelist(dfe, 'src', 'trg')

    # Now add node attributes to networkx objects
    #   these are taken from the Attribute file (df) created earlier
    # Create a present variable which indicates that these products are present in product space,
    # as not all products in product space are present in the data to be plotted (because we only plot products with more than >$40 million in trade)
    df_plot_dataframe['present'] = 1
    ATTRIBUTES = df_plot_attribute_cols + ['present']
    for ATTRIBUTE in ATTRIBUTES:
        dft = df_plot_dataframe[[df_plot_node_col, ATTRIBUTE]]
        dft['count'] = 1
        dft = dft.groupby([df_plot_node_col, ATTRIBUTE],
                          as_index=False)['count'].sum()
        # ** drop if missing , and drop duplicates
        dft.dropna(inplace=True)
        dft.drop(['count'], axis=1, inplace=True)
        dft.drop_duplicates(subset=[df_plot_node_col, ATTRIBUTE], inplace=True)
        dft.set_index(df_plot_node_col, inplace=True)
        dft_dict = dft[ATTRIBUTE].to_dict()
        for i in sorted(G.nodes()):
            try:
                G.node[i][ATTRIBUTE] = dft_dict[i]
            except Exception:
                G.node[i][ATTRIBUTE] = 'Missing'
        for i in sorted(G2.nodes()):
            try:
                G2.node[i][ATTRIBUTE] = dft_dict[i]
            except Exception:
                G2.node[i][ATTRIBUTE] = 'Missing'
    # Cross-check that attributes have been added correctly
    # nx.get_node_attributes(G2,'color')
    # nx.get_node_attributes(G,'color')

    # Create color + size lists that networkx can use when plotting
    groups = set(nx.get_node_attributes(G2, 'color').values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = G.nodes()
    nodes2 = G2.nodes()
    colorsl = [G.node[n]['color'] for n in nodes]
    colorsl2 = [G2.node[n]['color'] for n in nodes2]
    SIZE_VARIABLE = 'node_size'
    sizesl = [G.node[n][SIZE_VARIABLE] for n in nodes]

    # adjust value below to increase the PLOTTED size of nodes, depending on the resolution of the final plot
    # (e.g. if you want to zoom in into the product space and thus set a higher resolution, you may want to set this higher)
    sizesl2 = [G.node[n]['node_size'] * 350 for n in nodes]
    # Now draw the product space
    f = plt.figure(1, figsize=(20, 20))
    ax = f.add_subplot(1, 1, 1)
    # turn axes off
    plt.axis('off')
    # set white background
    f.set_facecolor('white')
    # now draw full product space in background, transparent with small node_size
    nx.draw_networkx(G2, nodes_pos, node_color='gray', ax=ax,
                     node_size=10, with_labels=False, alpha=0.1)
    # now draw the product space based on the data (e.g. trade data)
    nx.draw_networkx(G, nodes_pos, node_color=colorsl, ax=ax,
                     node_size=sizesl2, with_labels=False, alpha=1)
    # save file
    plt.savefig(output_dir_image)
    # show
    plt.show()
