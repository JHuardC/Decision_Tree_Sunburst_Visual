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

    """
    Takes a Decision Tree Model and produces a sunburst plot representing the 
    decision path of the training data through the model.

    The Decision Tree Model has the decision path of the training data extracted 
    to a structured array, which is then passed as the data_frame argument of 
    the original plotly express sunburst function. The structured array contains 
    the following fields which are mapped to the sunburst parameters:

        - parent_id -> parents
        - split_id -> ids
        - split_name -> names
        - value -> values
        - colour -> color

    A sunburst plot represents hierarchial data as sectors laid out over
    several levels of concentric rings.
    
    Parameters
    ----------
    model: sklearn.tree.BaseDecisionTree instance
        
    color_continuous_scale: list of str
        Strings should define valid CSS-colors This list is used to build a
        continuous color scale when the column denoted by `color` contains
        numeric data. Various useful color scales are available in the
        `plotly.express.colors` submodules, specifically
        `plotly.express.colors.sequential`, `plotly.express.colors.diverging`
        and `plotly.express.colors.cyclical`.
    range_color: list of two numbers
        If provided, overrides auto-scaling on the continuous color scale.
    color_continuous_midpoint: number (default `None`)
        If set, computes the bounds of the continuous color scale to have the
        desired midpoint. Setting this value is recommended when using
        `plotly.express.colors.diverging` color scales as the inputs to
        `color_continuous_scale`.
    color_discrete_sequence: list of str
        Strings should define valid CSS-colors. When `color` is set and the
        values in the corresponding column are not numeric, values in that
        column are assigned colors by cycling through `color_discrete_sequence`
        in the order described in `category_orders`, unless the value of
        `color` is a key in `color_discrete_map`. Various useful color
        sequences are available in the `plotly.express.colors` submodules,
        specifically `plotly.express.colors.qualitative`.
    color_discrete_map: dict with str keys and str values (default `{}`)
        String values should define valid CSS-colors Used to override
        `color_discrete_sequence` to assign a specific colors to marks
        corresponding with specific values. Keys in `color_discrete_map` should
        be values in the column denoted by `color`. Alternatively, if the
        values of `color` are valid colors, the string `'identity'` may be
        passed to cause them to be used directly.
    hover_name: str or int or Series or array-like
        Either a name of a column in `data_frame`, or a pandas Series or
        array_like object. Values from this column or array_like appear in bold
        in the hover tooltip.
    hover_data: list of str or int, or Series or array-like, or dict
        Either a list of names of columns in `data_frame`, or pandas Series, or
        array_like objects or a dict with column names as keys, with values
        True (for default formatting) False (in order to remove this column
        from hover information), or a formatting string, for example ':.3f' or
        '|%a' or list-like data to appear in the hover tooltip or tuples with a
        bool or formatting string as first element, and list-like data to
        appear in hover as second element Values from these columns appear as
        extra data in the hover tooltip.
    custom_data: list of str or int, or Series or array-like
        Either names of columns in `data_frame`, or pandas Series, or
        array_like objects Values from these columns are extra data, to be used
        in widgets or Dash callbacks for example. This data is not user-visible
        but is included in events emitted by the figure (lasso selection etc.)
    labels: dict with str keys and str values (default `{}`)
        By default, column names are used in the figure for axis titles, legend
        entries and hovers. This parameter allows this to be overridden. The
        keys of this dict should correspond to column names, and the values
        should correspond to the desired label to be displayed.
    title: str
        The figure title.
    template: str or dict or plotly.graph_objects.layout.Template instance
        The figure template name (must be a key in plotly.io.templates) or
        definition.
    width: int (default `None`)
        The figure width in pixels.
    height: int (default `None`)
        The figure height in pixels.
    branchvalues: str
        'total' or 'remainder' Determines how the items in `values` are summed.
        Whenset to 'total', items in `values` are taken to be valueof all its
        descendants. When set to 'remainder', itemsin `values` corresponding to
        the root and the branches:sectors are taken to be the extra part not
        part of thesum of the values at their leaves.
    maxdepth: int
        Positive integer Sets the number of rendered sectors from any given
        `level`. Set `maxdepth` to -1 to render all thelevels in the hierarchy.

    Returns
    -------
    plotly.graph_objects.Figure
    """

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
        data_frame = all_node_details,
        parents = 'parent_id', 
        ids = 'split_id', 
        names = 'split_name', 
        values = 'value', 
        color = 'colour',
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