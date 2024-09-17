from qiime2.plugin import (
    Plugin,
    Int,
    Float,
    Range,
    Metadata,
    Str,
    Bool,
    Choices,
    MetadataColumn,
    Categorical,
    List,
    Citations,
    TypeMatch,
    Numeric,
    SemanticType,
)

import csv
import skbio
from q2_types.feature_table import (
    FeatureTable,
    Composition,
    BIOMV210Format,
    BIOMV210DirFmt,
    Frequency,
    Design,
)
from q2_types.feature_data import TSVTaxonomyFormat, FeatureData, Taxonomy
import qiime2
import numpy as np
import biom
import zarr
import pandas as pd

import sys, os
from os.path import join, dirname
# q2_classo_dir = dirname(os.getcwd())
q2_classo_dir = dirname('/opt/project/')
sys.path.append(q2_classo_dir)
#print(sys.path)
#sys.path.remove('/opt/project/q2-classo')
import q2_classo as q2c

version = qiime2.__version__
# citations = Citations.load('citations.bib', package='q2_classo')


plugin = Plugin(
    name="classo",
    version="0.0.0.dev0",
    website="https://github.com/Leo-Simpson/q2-classo",
    package="q2-classo",
    short_description=(
        "Package for constrained sparse regression and classification"
    ),
    description=(
        "This is QIIME 2 plugin that enables sparse and robust "
        "linear regression and classification with linear "
        "equality constraints on the model parameters."
    ),
)


CLASSOProblem = SemanticType("CLASSOProblem")
ConstraintMatrix = SemanticType("ConstraintMatrix")
Weights = SemanticType("Weights")
# Design =
# SemanticType('Design', variant_of=FeatureTable.field['content'])


plugin.register_formats(
    q2c.CLASSOProblemDirectoryFormat,
    q2c.ConstraintDirectoryFormat, q2c.WeightsDirectoryFormat
)
plugin.register_semantic_type_to_format(
    ConstraintMatrix, artifact_format=q2c.ConstraintDirectoryFormat
)
plugin.register_semantic_type_to_format(
    Weights, artifact_format=q2c.WeightsDirectoryFormat
)

plugin.register_semantic_type_to_format(
    CLASSOProblem, artifact_format=q2c.CLASSOProblemDirectoryFormat
)

# plugin.register_semantic_type_to_format(
#   FeatureTable[Design],artifact_format=BIOMV210DirFmt)

plugin.register_semantic_types(CLASSOProblem, ConstraintMatrix, Weights)


# generate_data
plugin.methods.register_function(
    function=q2c.generate_data,
    inputs={"taxa": FeatureData[Taxonomy]},
    parameters={"n": Int, "d": Int, "d_nonzero": Int, "classification": Bool},
    outputs=[("x", FeatureTable[Design]), ("c", ConstraintMatrix)],
    input_descriptions={
        "taxa": "Taxonomy of the data. If it is given,"
        " it will generate random data associated to this"
    },
    parameter_descriptions={
        "n": "number of sample",
        "d": "number of features",
        "d_nonzero": "number of non nul componants in beta",
        "classification": (
            "boolean, if set to True," " y will be a vector with only -1 and 1"
        ),
    },
    output_descriptions={
        "x": "Matrix representing the data of the problem",
        "c": "Matrix representing the constraint of the problem",
    },
    name="generate_data",
    description=("Function that build random data"),
)


# features_clr
# we sue Design instead of Composition tyoe because other plugins neglect compositionality
# Design is somewhat compromise
plugin.methods.register_function(
    function=q2c.transform_features,
    inputs={"features": FeatureTable[Composition | Frequency | Design]},
    parameters={"transformation": Str, "coef": Float},
    outputs=[("x", FeatureTable[Design])],
    input_descriptions={
        "features": (
            "Matrix representing the compositional "
            "data of the problem, in order to clr transform it"
        )
    },
    parameter_descriptions={
        "transformation": (
            " String representing the name of the "
            "transformation we will use "
        ),
        "coef": (
            "Pseudocount that should be put instead of zeros"
            " in the feature table. Default value is 0.5"
        ),
    },
    output_descriptions={"x": "Matrix representing the data of the problem"},
    name="transform-features",
    description=(
        "Perform transformation, "
        "from FeatureTable[Composition/Frequency]"
        " prior to regress or classify,"
        " default transformation is centered log ratio"
    ),
)


# add_taxa
plugin.methods.register_function(
    function=q2c.add_taxa,
    inputs={
        "features": FeatureTable[Design],
        "weights": Weights,
        "taxa": FeatureData[Taxonomy],
    },
    parameters={},
    outputs=[("x", FeatureTable[Design]), ("aweights", Weights)],
    input_descriptions={
        "features": "Matrix representing the data of the problem",
        "weights": "Weights vector, default is the all one vector",
        "taxa": (
            "Taxonomic table in order to build matrix A"
            " and then change the problem to the new"
            " formulation (with log(X)A instead of log(X))"
        ),
    },
    parameter_descriptions={},
    output_descriptions={
        "x": "Matrix representing the data of the problem",
        "aweights": (
            "Weights vector rescaled"
            "by dividing it for each node"
            "by the number of leaves in the taxa tree"
        ),
    },
    name="add_taxa",
    description=("Function that change the data thanks to a taxonomic tree"),
)


# add_covariates
plugin.methods.register_function(
    function=q2c.add_covariates,
    inputs={"features": FeatureTable[Design], "c": ConstraintMatrix, "weights":Weights},
    parameters={"covariates": Metadata, "to_add": List[Str],"rescale":List[Bool], "w_to_add":List[Float]},
    outputs=[
        ("new_features", FeatureTable[Design]),
        ("new_c", ConstraintMatrix),
        ("new_w", Weights)
    ],
    input_descriptions={
        "features": "Matrix representing the data of the problem",
        "c": "Constraint matrix",
        "weights": "weigths that are gonna be updated"
    },
    parameter_descriptions={
        "covariates": (
            "output matrix, with several columns,"
            " including the one we want to regress on"
        ),
        "to_add": (
            "names of columns of y to add to the"
            " feature table after having normalized them"
        ),
        "rescale": "Either or not the columns should be rescaled",
        "w_to_add": "Regularization weights corresponding to new covariates"
    },
    output_descriptions={
        "new_features": "Feature table  with new columns taken from y",
        "new_c": "Updated matrix c, with 0 on the new added columns",
        "new_w":"Updated regularization weights "
    },
    name="add_covariates",
    description=(
        "Function to add some features that are in the metadata,"
        " or to build the feature matrix from metadata (tsv file)"
    ),
)


# regress
plugin.methods.register_function(
    function=q2c.regress,
    inputs={
        "features": FeatureTable[Design],
        "c": ConstraintMatrix,
        "weights": Weights,
        # 'taxa': FeatureData[Taxonomy]
    },
    parameters=q2c.regress_parameters,
    outputs=[("result", CLASSOProblem)],
    input_descriptions={
        "features": "Matrix representing the data of the problem",
        "c": "Constraint matrix, default is the zero-sum",
        "weights": "Vector of weights for penalization",
        # 'taxa':'Taxonomic table in order to build matrix A
        # and then change the problem to the new formulation
        # (with log(X)A instead of log(X))'
    },
    parameter_descriptions=q2c.regress_parameter_descriptions,
    output_descriptions={
        "result": (
            "Directory format that will contain all "
            "information about the problem solved"
        )
    },
    name="regress",
    description=(
        "The function computes the "
        "constrainted_sparse_regression vector "
        "with respect to the formulation of regression "
        "that is asked and with respect to the model selection"
        " parameters given"
    )
    # citations=[citations['Weiss2017']]
)

# classify
plugin.methods.register_function(
    function=q2c.classify,
    inputs={
        "features": FeatureTable[Design],
        "c": ConstraintMatrix,
        "weights": Weights,
        # 'taxa': FeatureData[Taxonomy]
    },
    parameters=q2c.classify_parameters,
    outputs=[("result", CLASSOProblem)],
    input_descriptions={
        "features": "Matrix representing the data of the problem",
        "c": "Constraint matrix, default is the zero-sum",
        "weights": "Vector of weights for penalization",
        # 'taxa':'Taxonomic table in order to build matrix A
        #  and then change the problem to the new formulation
        #  (with log(X)A instead of log(X))'
    },
    parameter_descriptions=q2c.classify_parameter_descriptions,
    output_descriptions={
        "result": (
            "Directory format that will"
            " contain all information about the problem solved"
        )
    },
    name="regress",
    description=(
        "The function computes the "
        "constrainted_sparse_regression vector "
        "with respect to the formulation of regression"
        " that is asked and with respect to the model"
        " selection parameters given"
    )
    # citations=[citations['Weiss2017']]
)


# predict
plugin.methods.register_function(
    function=q2c.predict,
    inputs={"features": FeatureTable[Design], "problem": CLASSOProblem},
    parameters={},
    outputs=[("predictions", CLASSOProblem)],
    input_descriptions={
        "features": "Matrix representing the data of the problem",
        "problem": (
            "output of regress or classify that contains"
            " the solutions : Directory format that will contain all "
            "information about the problem solved"
        ),
        # 'taxa':'Taxonomic table in order to build matrix A
        #  and then change the problem to the new formulation
        # (with log(X)A instead of log(X))'
    },
    parameter_descriptions={},
    output_descriptions={
        "predictions": (
            "Directory format that contains"
            " the predictions on each"
            " model selection computed"
        )
    },
    name="predict",
    description=(
        "The function computes a prediction of the output"
        " y with respect to the feature table inputed, "
        "and the model selection computed in problem"
    )
    # citations=[citations['Weiss2017']]
)


# summarize
plugin.visualizers.register_function(
    function=q2c.summarize,
    inputs={
        "problem": CLASSOProblem,
        "taxa": FeatureData[Taxonomy],
        "predictions": CLASSOProblem,
    },
    parameters={"maxplot": Int},
    input_descriptions={
        "problem": (
            "classo problem object containing "
            "the solution of the regression"
        ),
        "taxa": (
            "Taxonomic table in order to build matrix A"
            " and then change the problem to the new "
            "formulation (with log(X)A instead of log(X))"
        ),
        "predictions": (
            "classo problem object containing the"
            " predictions according to the "
            "model selections computed if "
            "the function predict has been used"
        ),
    },
    parameter_descriptions={
        "maxplot": (
            "maximum number of point in one bar plot"
            " ie maximum number of coefficient plotted"
            " in StabSel profile or in beta plots"
        )
    },
    name="summarize",
    description=(
        "Summarize the object created "
        "by the regression with its characteristics"
    ),
)


"""
Transformers
"""





@plugin.register_transformer
def _0(obj: q2c.classo_problem) -> q2c.CLASSOProblemDirectoryFormat:
    # for output of regress
    ff = q2c.CLASSOProblemDirectoryFormat()
    zipfile = str(ff.path / "problem.zip")
    store = zarr.ZipStore(zipfile, mode="w")
    root = zarr.open(store=store)
    q2c.to_zarr(obj, "problem", root)
    store.close()
    return ff


@plugin.register_transformer
def _1(obj: q2c.prediction_data) -> q2c.CLASSOProblemDirectoryFormat:
    ff = q2c.CLASSOProblemDirectoryFormat()
    zipfile = str(ff.path / "problem.zip")
    store = zarr.ZipStore(zipfile, mode="w")
    root = zarr.open(store=store)
    q2c.to_zarr(obj, "predictions", root)
    store.close()
    return ff


@plugin.register_transformer
def _2(ff: q2c.ZarrProblemFormat) -> zarr.hierarchy.Group:
    # for visualizers
    store = zarr.ZipStore(str(ff), mode="r")
    root = zarr.open(store=store)
    return root


@plugin.register_transformer
def _3(obj: np.ndarray) -> q2c.ConstraintDirectoryFormat:
    # for C in generate data, or generate constraint
    ff = q2c.ConstraintDirectoryFormat()
    filename = str(ff.path / "cmatrix.zip")
    zarr.save(filename, obj)
    return ff


@plugin.register_transformer
def _4(obj: q2c.ConstraintFormat) -> np.ndarray:
    # for C in regress
    return zarr.load(str(obj))


@plugin.register_transformer
def _5(ff: TSVTaxonomyFormat) -> skbio.TreeNode:
    # transformer for taxa tree
    root = skbio.TreeNode("root", length=0)
    line = 0
    with ff.open() as fh:
        reader = iter(csv.reader(fh, delimiter="\t"))
        next(reader)  # skip header
        for row in reader:
            id_, taxonomy = row[:2]
            taxonomy = taxonomy.split(";")
            node = root
            for taxon in taxonomy:
                new = True
                if taxon[0] == " ":
                    tax = taxon[1:]
                else:
                    tax = taxon

                for child in node.children:
                    if child.name == tax:
                        node = child
                        new = False
                        break
                if new:
                    child = skbio.TreeNode(tax, length=1)
                    node.append(child)
                    node = child

            # node.append(skbio.TreeNode('Beta%i'%line, length=1))
            node.append(skbio.TreeNode(id_, length=1))

            line += 1

    return root



@plugin.register_transformer
def _5(obj: np.ndarray) -> q2c.WeightsDirectoryFormat:
    # for C in generate data, or generate constraint
    ff = q2c.WeightsDirectoryFormat()
    filename = str(ff.path / "cmatrix.zip")
    zarr.save(filename, obj)
    return ff


@plugin.register_transformer
def _6(obj: q2c.WeightsFormat) -> np.ndarray:
    return zarr.load(str(obj))




"""
@plugin.register_transformer
def _2(obj : np.ndarray) -> BIOMV210Format :
    # for generate X
    ff = BIOMV210Format()
    l1 = [str(i) for i in range(len(obj[0]))]
    l2 = [str(i) for i in range(len(obj))]
    data = biom.Table(obj.T,observation_ids=l1,sample_ids=l2)
    with ff.open() as fh:
        data.to_hdf5(fh, generated_by='qiime2 %s' % version)
    return ff


@plugin.register_transformer
def _3(ff: BIOMV210Format) -> np.ndarray:
    # for X in regress
    with ff.open() as fh:
        table = biom.Table.from_hdf5(fh)
    array = table.matrix_data.toarray().T # numpy array
    sample_ids = table.ids(axis='sample')
    feature_ids = table.ids(axis='observation')
    data = pd.DataFrame(array, index=sample_ids, columns=feature_ids)
    return array
"""
