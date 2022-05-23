import random

import matplotlib.image as mpimg
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def plot_with_image(G):
    pos=nx.circular_layout(G)

    fig=plt.figure(figsize=(15,15))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)

    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    piesize=0.05  # this is the image size
    p2=piesize/2.0
    for n in G:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['img'])
        a.axis('off')
    ax.axis('off')
    plt.show()


def graph_augment(res, seq_length, sim_thresh, head_tail_length, argmax):

    # build nodes
    G = nx.DiGraph()
    for i in range(len(res)):
        G.add_node(i, seq=res[i])
        print(G.nodes[i]['seq'])
        plt.axis('off')
        plt.plot(res[i], color='black')
        plt.savefig('node_fig'+str(i)+'.png')
        img=mpimg.imread('node_fig'+str(i)+'.png')
        G.add_node(i, img=img)
        plt.clf()
        os.remove('node_fig'+str(i)+'.png')

    def sim(a, b):
        sum = 0
        for i in range(head_tail_length):
            sum += (np.linalg.norm(a[i] - b[i]))
        return 1 / sum

    def fit(a,b):
        c = []
        for i in range(len(a)):
            sigma_i = 1 / (1 + math.exp(-i))
            c.append((1 -sigma_i) * a[i] + sigma_i * b[i])
        return c

    def plot_graph(G):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
        plt.savefig('graph.png')

    # iterate over the nodes
    for u, u_att in G.nodes(data=True):
        list_neighbors = {}
        list_all = {}
        sum_sim = 0
        # find the edges weights
        for v, v_att in G.nodes(data=True):
            if u != v:
                sim_u_v = sim(u_att['seq'][-head_tail_length:], v_att['seq'][:head_tail_length])
                list_all[v] = sim_u_v
                if sim_u_v > sim_thresh:
                    list_neighbors[v] = sim_u_v

        # if no neighbors found
        if not bool(list_neighbors):
            max_value_keys = [key for key in list_all.keys() if
                              list_all[key] in sorted(list_all.values(), reverse=True)[:argmax]]
            for i in max_value_keys:
                list_neighbors[i] = list_all[i]

        # build the weighted edges
        for v in list_neighbors:
            G.add_edge(u, v, weight=(list_neighbors[v] - sim_thresh) / (sum_sim - sim_thresh))

    plot_graph(G)

    from random_walk import RandomWalk

    random_walk = RandomWalk(G, walk_length=walk_length, num_walks=1, p=1, q=1, weight_key= 'weight', workers=1)

    walklist = random_walk.walks

    # for w in walklist:
    #     print(w)

    w = walklist[0]
    generated_sequence = list(G.nodes[w[0]]['seq'])
    for i in range(1, len(w)):
        generated_sequence.extend(G.nodes[w[i]]['seq'])
        a_t = generated_sequence[int(-seq_length - head_tail_length/2 ) : - seq_length]
        b_t = generated_sequence[- seq_length : int(-seq_length + head_tail_length/2 )]
        generated_sequence[int(-seq_length - head_tail_length/2 ) : -seq_length] = fit(a_t, b_t)
        del generated_sequence[-seq_length: int( -seq_length + head_tail_length/2)]
        # print(G.nodes[w[i]]['seq'])

    return generated_sequence


res = []
seq_length = 8
sim_thresh = 1 / 2
walk_length = 10
head_tail_length = 4
argmax = 2
for i in range(10):
    res.append([random.randint(0, 22) for j in range(seq_length)])
# print(res)

generated_data = graph_augment(res, seq_length, sim_thresh, head_tail_length, argmax)
print(len(generated_data))