import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from itertools import count
import json
import networkx as nx


def create_product_space(df_plot_dataframe=None,
                         df_plot_node_col=None,
                         df_node_size_col=None):

    # No legend, not properly coded yet
    show_legend = 0

    # Copy dataframe so original won't be overwritten
    df_plot =  df_plot_dataframe.copy()

    NORMALIZE_NODE_SIZE = 1
    if NORMALIZE_NODE_SIZE == 1:

        """
        The distribution of export values is highly skewed, which makes it hard to visualize
        them properly (certain products will overshadow the rest of the network).
        We create a new size column below in which we normalize the export values.
        """

        ### Normalize node size (0.1 to 1)
        def normalize_col(dft,col,minsize=0.1,maxsize=1):
            """
            Normalizes column values with largest and smallest values capped at min at max
            For use in networkx

            returns pandas column
            """

            alpha = maxsize-minsize
            Xl = dft[dft[col]>0][col].quantile(0.10)
            Xh = dft[dft[col]>0][col].quantile(0.95)
            dft['node_size'] = 0
            dft.loc[ dft[col]>=Xh,'node_size'] = maxsize
            dft.loc[ (dft[col]<=Xl) & (dft[col]!=0),'node_size'] = minsize
            dft.loc[ (dft[col]<Xh) & (dft[col]>Xl),'node_size'] = ((alpha*(dft[col]-Xl))/(Xh-Xl))+(1-alpha)
            dft.loc[ (dft[col]<Xh) & (dft[col]>0),'node_size'] = ((alpha*(dft[col]-Xl))/(Xh-Xl))+(1-alpha)

            return dft['node_size']

        df_plot['node_size'] = normalize_col(df_plot,df_node_size_col,minsize=0.1,maxsize=1)

    ADD_COLORS_ATLAS = 1
    if ADD_COLORS_ATLAS == 1:

        # First add product codes from original file (full strings were used for illustrative purposes above but we need the actual codes to merge data from other sources, e.g. node colors)
        df_plot = pd.merge(df_plot,df_orig[['product_name','product_code']].drop_duplicates(),how='left',on='product_name')
        dft = pd.read_csv('https://www.dropbox.com/s/rlm8hu4pq0nkg63/hs4_hex_colors_intl_atlas.csv?dl=1')

        # Transform product_code into int (accounts for missing in pandas, if necessary)
        # keep only numeric product_codes (this drops 'unspecified' as well as services for now;
        # - as the latter needs a separate color classification)
        df_plot = df_plot[df_plot['product_code'].astype(str).str.isnumeric()]
        # -- also drop 9999 product code; unknown
        df_plot = df_plot[df_plot['product_code'].astype(str)!='9999']
        # -- to allow merge, rename and transform both variables into int
        dft['hs4'] = dft['hs4'].astype(int)
        df_plot['product_code'] = df_plot['product_code'].astype(int)
        if 'color' in df_plot.columns:
            df_plot.drop(['color'],axis=1,inplace=True,errors='ignore')
        df_plot = pd.merge(df_plot,dft[['hs4','color']],how='left',left_on='product_code',right_on='hs4')
        # drop column merged from dft
        df_plot.drop(['hs4'],axis=1,inplace=True,errors='ignore')

    ADD_NODE_POSITIONS_ATLAS = 1
    if ADD_NODE_POSITIONS_ATLAS == 1:

        # Load position of nodes (x, y coordinates of nodes from original Atlas file)
        import urllib.request, json
        with urllib.request.urlopen("https://www.dropbox.com/s/r601tjoulq1denf/network_hs92_4digit.json?dl=1") as url:
            networkjs = json.loads(url.read().decode())

    CREATE_NETWORKX_OBJECT_AND_PLOT = 1
    if CREATE_NETWORKX_OBJECT_AND_PLOT == 1:

        # Convert json into python list and dictionary
        nodes = []
        nodes_pos = {}
        for x in networkjs['nodes']:
            nodes.append(int(x['id']))
            nodes_pos[int(x['id'])] = (int(x['x']),-int(x['y']/1.5))

        # Define product space edge list (based on strength from the json)
        edges = []
        for x in networkjs['edges']:
            if x['strength'] > 1 or 1 == 1:
                edges.append((int(x['source']),int(x['target'])))
        dfe = pd.DataFrame(edges)
        dfe.rename(columns={0: 'src'}, inplace=True)
        dfe.rename(columns={1: 'trg'}, inplace=True)

        # Only select edges of nodes that are also present in product space
        dfe2 = pd.DataFrame(np.append(dfe['src'].values,dfe['trg'].values)) # (some products may not be in there)
        dfe2.drop_duplicates(inplace=True)
        dfe2.rename(columns={0: 'node'}, inplace=True)
        dfn2 = pd.merge(df_plot,dfe2,how='left',left_on=df_plot_node_col,right_on='node',indicator=True)

        # Drop products from this dataframe that are not in product space
        dfn2 = dfn2[dfn2['_merge']=='both']

        # Create networkx objects in Python

        # G object = products that will be plotted
        G=nx.from_pandas_edgelist(dfn2,'product_code','product_code')

        # G2 object = all nodes and edges from the original product space
        # - Those that are not plotted will be gray in the background,
        # - e.g. products for which there is no info
        G2=nx.from_pandas_edgelist(dfe,'src','trg')

        # Add node attributes to networkx objects
        # - Create a 'present' variable which indicates that these products are present in product space,
        # - as not all products in product space are present in the data to be plotted
        # - (e.g. because we could filter only to plot products with more than >$40 million in trade)
        df_plot['present'] = 1
        ATTRIBUTES = ['node_size'] + ['color'] + ['present']
        for ATTRIBUTE in ATTRIBUTES:
            dft = df_plot[[df_plot_node_col,ATTRIBUTE]]
            dft['count'] = 1
            dft = dft.groupby([df_plot_node_col,ATTRIBUTE],as_index=False)['count'].sum()
            #** drop if missing , and drop duplicates
            dft.dropna(inplace=True)
            dft.drop(['count'],axis=1,inplace=True)
            dft.drop_duplicates(subset=[df_plot_node_col,ATTRIBUTE],inplace=True)
            dft.set_index(df_plot_node_col,inplace=True)
            dft_dict = dft[ATTRIBUTE].to_dict()
            for i in sorted(G.nodes()):
                try:
                    #G.node[i][ATTRIBUTE] = dft_dict[i]
                    G.nodes[i][ATTRIBUTE] = dft_dict[i]
                except Exception:
                    #G.node[i][ATTRIBUTE] = 'Missing'
                    G.nodes[i][ATTRIBUTE] = 'Missing'
            for i in sorted(G2.nodes()):
                try:
                    #G2.node[i][ATTRIBUTE] = dft_dict[i]
                    G2.nodes[i][ATTRIBUTE] = dft_dict[i]
                except Exception:
                    #G2.node[i][ATTRIBUTE] = 'Missing'
                    G2.nodes[i][ATTRIBUTE] = 'Missing'

        # Cross-check that attributes have been added correctly
        # nx.get_node_attributes(G2,df_color)
        # nx.get_node_attributes(G,df_color)

        # Create color + size lists which networkx uses for plotting
        groups = set(nx.get_node_attributes(G2,'color').values())
        mapping = dict(zip(sorted(groups),count()))
        nodes = G.nodes()
        nodes2 = G2.nodes()
        #colorsl = [G.node[n]['color'] for n in nodes]
        colorsl = [G.nodes[n]['color'] for n in nodes]
        #colorsl2 = [G2.node[n]['color'] for n in nodes2]
        colorsl2 = [G2.nodes[n]['color'] for n in nodes2]
        SIZE_VARIABLE = 'node_size'
        #sizesl = [G.node[n][SIZE_VARIABLE] for n in nodes]
        sizesl = [G.nodes[n][SIZE_VARIABLE] for n in nodes]
        # Adjust value below to increase the PLOTTED size of nodes, depending on the resolution of the final plot
        # (e.g. if you want to zoom in into the product space and thus set a higher resolution, you may want to set this higher)
        #sizesl2 = [G.node[n]['node_size']*350 for n in nodes]
        sizesl2 = [G.nodes[n]['node_size']*350 for n in nodes]

        # Create matplotlib object to draw the product space
        f = plt.figure(1,figsize=(20,20))
        ax = f.add_subplot(1,1,1)

        # turn axes off
        plt.axis('off')

        # set white background
        f.set_facecolor('white')

        # draw full product space in background, transparent with small node_size
        nx.draw_networkx(G2,nodes_pos, node_color='gray',ax=ax,node_size=10,with_labels=False,alpha=0.1)

        # draw the data fed in to the product space
        nx.draw_networkx(G,nodes_pos, node_color=colorsl,ax=ax,node_size=sizesl2,with_labels=False,alpha=1)

        # save file
        # plt.savefig(output_dir_image)

        # show the plot
        plt.show()
