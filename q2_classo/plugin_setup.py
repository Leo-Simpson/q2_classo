from qiime2.plugin import (Plugin, Int, Float, Range, Metadata, Str, Bool,
     Choices, MetadataColumn, Categorical, List,
     Citations, TypeMatch, Numeric, SemanticType)

import csv
import skbio
from q2_types.feature_table import FeatureTable, Composition, BIOMV210Format
from q2_types.feature_data import TSVTaxonomyFormat, FeatureData, Taxonomy
import qiime2
from . import  *
import numpy as np
import biom
import zarr
import pandas as pd


version=qiime2.__version__
#citations = Citations.load('citations.bib', package='q2_classo') 



plugin = Plugin(
name='classo',
                version='0.0.0.dev0',
website='https://github.com/Leo-Simpson/q2-classo',
package='q2-classo',
short_description=('Package for constrained sparse regression and classification'),
description=('This is QIIME 2 plugin that enables sparse and robust linear regression and classification with linear equality constraints on the model parameters.')
)


CLASSOProblem    = SemanticType("CLASSOProblem")
ConstraintMatrix = SemanticType("ConstraintMatrix")


plugin.register_formats(CLASSOProblemDirectoryFormat,ConstraintDirectoryFormat)
plugin.register_semantic_type_to_format(ConstraintMatrix,
                                        artifact_format=ConstraintDirectoryFormat)
plugin.register_semantic_type_to_format(CLASSOProblem, 
                                        artifact_format=CLASSOProblemDirectoryFormat)




plugin.methods.register_function(
           function=regress,
           inputs={'features': FeatureTable[Composition], 
                    'c':ConstraintMatrix,
                    'taxa': FeatureData[Taxonomy]
                    },
           parameters=regress_parameters,
           outputs= [('result',CLASSOProblem)],
           input_descriptions={'features': 'Matrix representing the data of the problem',
                                'c': 'Constraint matrix, default is the zero-sum',
                                'taxa':'Taxonomic table in order to build matrix A and then change the problem to the new formulation (with log(X)A instead of log(X))'
                                },
           parameter_descriptions=regress_parameter_descriptions,
           output_descriptions= {
               'result':"Directory format that will contain all information about the problem solved"
               },
           name='regress',
           description=("The function computes the constrainted_sparse_regression vector with respect to the formulation of regression that is asked and with respect to the model selection parameters given")
           #citations=[citations['Weiss2017']]
           )

plugin.methods.register_function(
           function=classify,
           inputs={'features': FeatureTable[Composition], 
                    'c':ConstraintMatrix,
                    'taxa': FeatureData[Taxonomy]
                    },
           parameters=classify_parameters,
           outputs= [('result',CLASSOProblem)],
           input_descriptions={'features': 'Matrix representing the data of the problem',
                                'c': 'Constraint matrix, default is the zero-sum',
                                'taxa':'Taxonomic table in order to build matrix A and then change the problem to the new formulation (with log(X)A instead of log(X))'
                                },
           parameter_descriptions=classify_parameter_descriptions,
           output_descriptions= {
               'result':"Directory format that will contain all information about the problem solved"
               },
           name='regress',
           description=("The function computes the constrainted_sparse_regression vector with respect to the formulation of regression that is asked and with respect to the model selection parameters given")
           #citations=[citations['Weiss2017']]
           )

plugin.methods.register_function(
           function=transform,
           inputs={'features': FeatureTable[Composition], 
                    'c': ConstraintMatrix
                    },
           parameters={'y': Metadata, 'to_add': List[Str]},
           outputs= [('new_x',FeatureTable[Composition]), ('new_c', ConstraintMatrix)],
           input_descriptions={'features': 'Matrix representing the data of the problem' , 
                                'c':'Constraint matrix'
                                },
           parameter_descriptions={'y': 'output matrix, with several columns, including the one we want to regress on',
                                    'to_add': 'names of columns of y to add to the feature table after having normalized them'},
           output_descriptions= {
               'new_x': 'Feature table  with new columns taken from y',
               'new_c':'Updated matrix c, with 0 on the new added columns'
               },
           name='regress',
           description=("The function computes the constrainted_sparse_regression vector with respect to the formulation of regression that is asked and with respect to the model selection parameters given")
           #citations=[citations['Weiss2017']]
           )




plugin.methods.register_function(
           function=generate_data,
           inputs={'taxa':FeatureData[Taxonomy]},
           parameters={'n':Int, 'd':Int, 'd_nonzero':Int, 'classification': Bool},
           outputs= [('x',FeatureTable[Composition]),('c',ConstraintMatrix)],
           input_descriptions={'taxa' : 'Taxonomy of the data. If it is given, it will generate random data associated to this'},
           parameter_descriptions={'n': 'number of sample', 'd': 'number of features','d_nonzero': 'number of non nul componants in beta' , 'classification' : 'boolean, if set to True, y will be a vector with only -1 and 1'},
           output_descriptions= {'x': 'Matrix representing the data of the problem','c':'Matrix representing the constraint of the problem'},
           name='generate_data',
           description=("Function that build random data")
           )


plugin.visualizers.register_function(
    function=summarize,
    inputs={'problem':CLASSOProblem},
    parameters={},
    input_descriptions={'problem': 'classo problem object containing the solution of the regression'},
    parameter_descriptions={},
    name='Summarize regression solution',
    description=('Summarize the object created by the regression with its characteristics')
)









'''
Transformers
'''

@plugin.register_transformer
def _0(obj: classo_problem) -> CLASSOProblemDirectoryFormat :
    # for output of regress
    ff = CLASSOProblemDirectoryFormat()
    zipfile = str(ff.path/'problem.zip')
    store = zarr.ZipStore(zipfile,mode='w')
    root = zarr.open(store=store)
    to_zarr(obj,'problem',root)
    store.close()
    return ff 

@plugin.register_transformer
def _1(ff : ZarrProblemFormat) -> zarr.hierarchy.Group : 
    # for visualizers
    store = zarr.ZipStore(str(ff),mode='r')
    root = zarr.open(store=store)
    return root




@plugin.register_transformer
def _2(obj : np.ndarray) -> ConstraintDirectoryFormat :
    # for C in generate data, or generate constraint
    ff = ConstraintDirectoryFormat()
    filename = str(ff.path/'cmatrix.zip')
    zarr.save(filename, obj)
    return ff


@plugin.register_transformer
def _3(obj : ConstraintFormat) -> np.ndarray : 
    # for C in regress
    return zarr.load(str(obj))



@plugin.register_transformer
def _4(ff: TSVTaxonomyFormat) -> skbio.TreeNode:
    # transformer for taxa tree 
    root = skbio.TreeNode('root', length=0)
    line = 0
    with ff.open() as fh:
        reader = iter(csv.reader(fh, delimiter='\t'))
        next(reader)  # skip header
        for row in reader:
            id_, taxonomy = row[:2]
            taxonomy = taxonomy.split(';')
            node = root
            for taxon in taxonomy:
                new = True
                if taxon[0]==' ': tax = taxon[1:]
                else : tax = taxon

                for child in node.children:
                    if child.name == tax:
                        node = child
                        new = False
                        break
                if new :
                     child = skbio.TreeNode(tax, length=1)
                     node.append(child)
                     node = child
                

            node.append(skbio.TreeNode('Beta%i'%line, length=1))
            line += 1

    return root


'''
@plugin.register_transformer
def _2(obj : np.ndarray) -> BIOMV210Format :
    # for generate X
    ff = BIOMV210Format()
    l1, l2 = [str(i) for i in range(len(obj[0]))],[str(i) for i in range(len(obj))]
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
'''
