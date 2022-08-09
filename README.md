# Decision_Tree_Sunburst_Visual

This repo contains a function that passes a trained sci-kit learn decision tree model and returns a plotly sunburst figure, based on the decision path of the training data. The sunburst produced is an interactive tool, useful for data mining/exploratory data analysis; it shows feature importances and can help provide insights into non-linear feature interations as described here: [Exploratory Data Mining with Classification and Regression Trees (CART)](https://www.apa.org/science/about/psa/2018/04/classification-regression-trees).

To clone this repository:

```git
git clone https://github.com/JHuardC/Decision_Tree_Sunburst_Visual.git
```

## Example

The following example uses a subset of diamond data, sourced from the following Kaggle data set: <https://www.kaggle.com/datasets/shivam2503/diamonds>

The feature "cut" of the diamond data is used as the target feature. The diamond data set was filtered to only "fair" and "good" cuts, making this a binary classification scenario.

Sci-kit learn's decision tree classifier was used to model the "cut" feature, see sample code provided for hyperparameters:

```python
# split diamonds
X_cols = [el for el in diamonds.columns if el != 'cut']

train_X,test_X,train_y,test_y = train_test_split(
    diamonds[X_cols],
    diamonds['cut'],
    test_size = 0.2,
    random_state = 42
)
    
# build a basic decision tree classifier
# initialize decision tree
tree = DecisionTreeClassifier(max_depth = 3)
# train
model = tree.fit(train_X, train_y)

# Create sunburst
fig = ts.visualize_tree_as_sunburst(model = model)
```

### Decision Tree of diamond data
![Example Decision Tree Sunburst](/figures/example_sunburst.png)
