# -*- coding: utf-8 -*-

import numpy as np
import pyvisgraph as vg


def construction_biased_tree(tree, n_max):
    """
    construction of the biased tree
    :param tree: biased tree
    :param n_max: maximum number of iterations
    :return: found path
    """

    for n in range(n_max):
        if n%200 == 0:
            print("Iterations done: %d" %(n))
        # biased sample
        x_new, x_angle, q_p_closest, label, target_b_state = tree.biased_sample()
        # couldn't find x_new
        if not x_new: continue
        # label of x_new
        # label = tree.task.get_label_landmark(x_new, tree.workspace_instance)
        # near state
        if tree.lite:
            # avoid near
            near_nodes = [q_p_closest]
        else:
            near_nodes = tree.near(tree.mulp2single(x_new))
            near_nodes = near_nodes + [q_p_closest] if q_p_closest not in near_nodes else near_nodes

        # check the line is obstacle-free: returns True if path is possible
        obs_check = tree.obstacle_check(near_nodes, x_new, label)
        # not obstacle-free
        if tree.lite and not list(obs_check.items())[0][1]: continue

        # iterate over each buchi state
        for b_state in tree.buchi.buchi_graph.nodes:
            # new product state
            q_new = (x_new, b_state)
            # extend
        
            added = tree.extend(q_new, x_angle, near_nodes, label, obs_check, target_b_state)
            # rewire
            if not tree.lite and added:
                tree.rewire(q_new, near_nodes, obs_check)

        # detect the first accepting state
        if len(tree.goals): break

    return tree.find_path(tree.goals)
