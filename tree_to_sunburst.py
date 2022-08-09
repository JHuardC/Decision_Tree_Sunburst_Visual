# -*- coding: utf-8 -*-
""" A function to take a trained decision tree and present the structure as a sunburst.

Recommended: set plotly io renderers to 'browser'.

The sunburst visual shows the decision path of the training data through the tree.

Useful for exploratory data analysis and Feature engineering/creation: The decision 
tree and sunburst visual can help users to understand the structure of the data by 
uncovering non-linear interatctions between features. A more detailed explanation of 
this technique can be found here: 

https://www.apa.org/science/about/psa/2018/04/classification-regression-trees

Created on Wed Sep 22 13:53:21 2021

@author: JHuardC
"""

from sklearn.tree import BaseDecisionTree
from plotly.graph_objects import Figure
import numpy as np
from collections import defaultdict
import plotly.express as px

def visualize_tree_as_sunburst(
    model: BaseDecisionTree,
    color_continuous_scale=None, 
    range_color=None, 
    color_continuous_midpoint=None, 
    color_discrete_sequence=None, 
    color_discrete_map=None, 
    hover_name=None, 
    hover_data=None, 
    custom_data=None, 
    labels=None, 
    title=None, 
    template=None, 
    width=None, 
    height=None, 
    branchvalues = 'total', 
    maxdepth=None
    ) -> Figure:

    # this list will be used to create a structured array of arguments
    sunburst_args = [
        ('parent_id', 'U256'),
        ('split_id', 'U256'),
        ('split_name', 'U256'),
        ('value', 'i'), 
        ('colour', 'U256')
    ]

    feature_names_lookup = defaultdict(lambda: '', enumerate(model.feature_names_in_))
    tree_node_structure = model.tree_.__getstate__()['nodes']

    split_name_templates = {'left_child': '{} <= {:.2f}', 'right_child': '{} > {:.2f}'}

    split_names_lookup = dict()
    split_value_lookup = dict()

    child_parent_lookup = defaultdict(lambda: '') # populated in loop
    all_node_details = [] # populated in loop
    for node_idx, node_info in enumerate(tree_node_structure):

        # check node is not a leaf node
        if node_info['left_child'] == -1:
            pass

        else:
            ### get node details

            # get split_name for greater than and less than or equal to sections
            feature = feature_names_lookup[node_info['feature']]
            threshold = node_info['threshold']
            left_idx = len(split_names_lookup)

            right_idx = str(left_idx + 1)
            left_idx = str(left_idx)
            
            split_names_lookup.update(
                {
                    left_idx: split_name_templates['left_child']\
                        .format(feature, threshold), 
                    right_idx: split_name_templates['right_child']\
                        .format(feature, threshold)
                }
            )

            # get the size of each partition
            split_value_lookup.update(
                {
                    left_idx: 
                        tree_node_structure[node_info['left_child']]['n_node_samples'], 
                    right_idx: 
                        tree_node_structure[node_info['right_child']]['n_node_samples']
                }
            )

            for el in [left_idx, right_idx]:

                node_details = [] # populate with data for sunburst_args

                # get parent_id
                node_details.append(child_parent_lookup[node_idx])

                # get split_id
                node_details.append(el)

                # get split_name
                node_details.append(split_names_lookup[el])

                # get value
                node_details.append(split_value_lookup[el])

                # get colour
                node_details.append(feature)

                # create structured array and add to all_node_details
                all_node_details.append(
                    np.array([tuple(node_details)], dtype = sunburst_args)
                )

            
            ### populate child_parent_lookup
            child_parent_lookup[node_info['left_child']] = left_idx
            child_parent_lookup[node_info['right_child']] = right_idx

    all_node_details = np.concatenate(all_node_details)

    fig = px.sunburst(
        parents = all_node_details['parent_id'], 
        ids = all_node_details['split_id'], 
        names = all_node_details['split_name'], 
        values = all_node_details['value'], 
        color = all_node_details['colour'],
        branchvalues = branchvalues,
        color_continuous_scale = color_continuous_scale, 
        range_color = range_color, 
        color_continuous_midpoint = color_continuous_midpoint, 
        color_discrete_sequence = color_discrete_sequence, 
        color_discrete_map = color_discrete_map, 
        hover_name = hover_name, 
        hover_data = hover_data, 
        custom_data = custom_data, 
        labels = labels, 
        title = title, 
        template = template, 
        width = width, 
        height = height,
        maxdepth = maxdepth
        )

    return fig