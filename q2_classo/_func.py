import numpy as np
import zarr
from classo import *

from q2_types.feature_table import FeatureTable, Composition
from q2_types.feature_data import FeatureData
import qiime2
import pandas as pd
import skbio
from ._tree import tree_to_matrix


# TODO : change C type to pandas dataframe


class prediction_data:
    def __init__(self, PATH=False, CV=False, StabSel=False, LAMfixed=False):
        self.PATH = PATH
        self.CV = CV
        self.StabSel = StabSel
        self.LAMfixed = LAMfixed


def generate_data(
        taxa: skbio.TreeNode = None,
        n: int = 100,
        d: int = 80,
        d_nonzero: int = 5,
        classification: bool = False,
) -> (pd.DataFrame, np.ndarray):
    label = np.array(
        ["A" + str(i) for i in range(d // 2)]
        + ["B" + str(i) for i in range(d - d // 2)]
    )
    label_gamma = label[:]
    A = None
    if taxa is not None:
        label2 = np.array([tip.name for tip in taxa.tips()])
        if len(label2) >= d:
            label = label2[:d]
        else:
            label[: len(label2)] = label2
        A, label_gamma = tree_to_matrix(taxa, label)

    (X, C, y), sol = random_data(
        n,
        d,
        d_nonzero,
        0,
        0.5,
        zerosum=True,
        seed=None,
        exp=True,
        A=A,
        classification=classification,
    )

    print(label_gamma[sol != 0])
    if classification:
        y = y == 1.0

    dfx = pd.DataFrame(
        data=X, index=[str(i) for i in range(len(X))], columns=label
    )
    pd.DataFrame(data={"id": range(len(y)), "col": y}).to_csv(
        "randomy.tsv", sep="\t", index=False
    )
    return dfx, C


def transform_features(
    features: pd.DataFrame, transformation: str = "clr", coef: float = 0.5
) -> pd.DataFrame:
    if transformation == "clr":
        X = features.values
        null_set = X <= 0.0# just ignore zero replacement for the sake of experiment
        X[null_set] = coef
        X = np.log(X)
        X = (X.T - np.mean(X, axis=1)).T
        # X = (X - np.mean(X, axis=0)) if X is (p,n)

        return pd.DataFrame(
            data=X, index=list(features.index.astype(str)),
            columns=list(features.columns.astype(str))
        )

    else:
        raise ValueError(
            "Unknown transformation name, use clr and not %r" % transformation
        )


def add_taxa(
        features: pd.DataFrame, weights: np.ndarray = None, taxa: skbio.TreeNode = None
) -> (pd.DataFrame, np.ndarray):
    X = features.values
    label = list(features.columns)
    A, label_new = tree_to_matrix(taxa, label, with_repr=True)
    nleaves = np.sum(A, axis=0)
    X_new = X.dot(A) / nleaves
    if weights is None:
        w_new = 1. / nleaves
    else:
        w_new = weights / nleaves
    dfx = pd.DataFrame(
        data=X_new, index=list(features.index), columns=label_new
    )

    return dfx, w_new


def add_covariates(
        covariates: qiime2.Metadata,
        to_add: str,
        features: pd.DataFrame = None,
        c: np.ndarray = None,
        rescale: list = None,
        weights: np.ndarray = None,
        w_to_add: list = None
) -> (pd.DataFrame, np.ndarray, np.ndarray):
    d = len(features.columns)
    covariates_df = covariates.to_dataframe()
    covariates_df = covariates_df[to_add]

    if rescale is None:
        rescale = [False] * len(to_add)
    elif len(rescale) != len(to_add):
        raise ValueError("List rescale and to_add have different lengths")

    if weights is None:
        weights = np.ones(d)
    elif len(weights) != d:
        raise ValueError("weigths and features have different dimension")

    if w_to_add is None:
        w_to_add = [1.] * len(to_add)
    elif len(w_to_add) != len(to_add):
        raise ValueError("Lists to_add and w_to_add have different lengths")

    for i, name in enumerate(to_add):
        type_ = covariates.columns[name].type
        serie = covariates_df[name]
        vect = serie.to_numpy()
        if type_ == "numeric":
            # if vect has a NaN ==> raise error
            if rescale[i]:
                vect = vect - np.mean(vect)
                vect = vect / np.linalg.norm(vect)
            missing = np.isnan(vect)
            vect[missing] = np.mean(vect[~missing])
            covariates_df[name] = vect
            weights = np.append(weights, w_to_add[i])
        elif type_ == "categorical":
            values_encont = []
            for value_name in vect:
                if value_name not in values_encont:
                    values_encont.append(value_name)
                    binary_vect = 1 * (vect == value_name)
                    nb_col = len(covariates_df.columns)
                    covariates_df.insert(nb_col - 1, column=name + ' = ' + value_name, value=binary_vect)
                    weights = np.append(weights, w_to_add[i])

            covariates_df = covariates_df.drop(columns=name)

    if features is not None:
        # features, covariates_df
        # = features.align(covariates_df, join='inner',axis=0)
        X = pd.concat([features, covariates_df], axis=1, join="inner")
        d_new = len(X.columns)

        if c is None:
            c = np.ones((1, d))

        c_new = np.zeros((len(c), d_new))
        c_new[:, : d] = c
    else:
        X = covariates_df
        c_new = np.zeros((1, d_new))

    return X, c_new, weights


def regress(
        features: pd.DataFrame,
        y: qiime2.NumericMetadataColumn,
        c: np.ndarray = None,
        weights: np.ndarray = None,
        do_yshift: bool = False,
        # taxa: skbio.TreeNode = None,
        # PATH parameters :
        path: bool = True,
        path_numerical_method: str = "not specified",
        path_n_active: int = 0,
        path_nlam_log: int = 40,
        path_lamin_log: float = 1e-2,
        # CV parameters :
        cv: bool = True,
        cv_numerical_method: str = "not specified",
        cv_seed: int = 1,
        cv_one_se: bool = True,
        cv_subsets: int = 5,
        cv__nlam: int = 100,
        cv_lamin: float = 1e-3,
        cv_logscale: bool = True,
        # StabSel parameters :
        stabsel: bool = True,
        stabsel_numerical_method: str = "not specified",
        stabsel_seed: int = None,  # do something here ! for now it can be a bool !
        stabsel_lam: float = -1.0,  # if negative, then it means 'theoretical'
        stabsel_true_lam: bool = True,
        stabsel_method: str = "first",
        stabsel_b: int = 50,
        stabsel_q: int = 10,
        stabsel_percent_ns: float = 0.5,
        stabsel_lamin: float = 1e-2,
        stabsel_threshold: float = 0.7,
        stabsel_threshold_label: float = 0.4,
        # might unneeded here, but needed for visualisation
        # LAMfixed parameters :
        lamfixed: bool = True,
        lamfixed_numerical_method: str = "not specified",
        lamfixed_lam: float = -1.0,  # if negative, then it means 'theoretical'
        lamfixed_true_lam: bool = True,
        # Formulation parameters
        concomitant: bool = True,
        huber: bool = False,
        rho: float = 1.345,
        intercept: bool = True,
) -> classo_problem:
    complete_y = y.to_series()
    complete_y = complete_y[~complete_y.isna()]

    features, pdY = features.align(y.to_series(), join="inner", axis=0)
    missing = pdY.isna()
    training_labels = list(pdY[~missing].index)
    label_missing = list(pdY.index[missing])
    if label_missing:
        print("{} are missing in y ".format(label_missing))
    Y = pdY[~missing].to_numpy()
    X = features.values[~missing, :]

    print(Y.shape, X.shape)

    if do_yshift:
        Y = Y - np.mean(Y)

    problem = classo_problem(X, Y, C=c, label=list(features.columns))
    problem.formulation.huber = huber
    problem.formulation.concomitant = concomitant
    problem.formulation.rho = rho
    problem.formulation.intercept = intercept
    d = X.shape[1]
    if weights is not None:
        if len(weights) < d:
            problem.formulation.w = np.concatenate([weights, np.ones(d - len(weights))], axis=0)
        else:
            problem.formulation.w = weights[:d]

    problem.model_selection.PATH = path
    if path:
        param = problem.model_selection.PATHparameters
        param.numerical_method = path_numerical_method
        param.n_active = path_n_active
        param.logscale = True
        param.Nlam = path_nlam_log
        param.lamin = path_lamin_log

    problem.model_selection.CV = cv
    if cv:
        param = problem.model_selection.CVparameters
        param.numerical_method = cv_numerical_method
        param.seed = cv_seed
        param.oneSE = cv_one_se
        param.Nsubsets = cv_subsets
        param.lamin = cv_lamin
        param.Nlam = cv__nlam
        param.logscale = cv_logscale

    problem.model_selection.StabSel = stabsel
    if stabsel:
        param = problem.model_selection.StabSelparameters
        param.numerical_method = stabsel_numerical_method
        param.seed = stabsel_seed
        param.true_lam = stabsel_true_lam
        param.method = stabsel_method
        param.B = stabsel_b
        param.q = stabsel_q
        param.percent_nS = stabsel_percent_ns
        param.lamin = stabsel_lamin
        param.threshold = stabsel_threshold
        param.threshold_label = stabsel_threshold_label
        if stabsel_lam > 0.0:
            param.lam = stabsel_lam
        else:
            param.lam = "theoretical"

    problem.model_selection.LAMfixed = lamfixed
    if lamfixed:
        param = problem.model_selection.LAMfixedparameters
        param.numerical_method = lamfixed_numerical_method
        param.true_lam = lamfixed_true_lam
        if lamfixed_lam > 0.0:
            param.lam = lamfixed_lam
        else:
            param.lam = "theoretical"

    print("start solve !")
    problem.solve()
    print("finished solve ! ")

    problem.data.complete_y = complete_y.values
    problem.data.complete_labels = list(complete_y.index)
    problem.data.training_labels = training_labels

    return problem


def classify(
        features: pd.DataFrame,
        y: qiime2.CategoricalMetadataColumn,
        c: np.ndarray = None,
        weights: np.ndarray = None,
        # taxa: skbio.TreeNode = None,
        # PATH parameters :
        path: bool = True,
        path_numerical_method: str = "not specified",
        path_n_active: int = 0,
        path_nlam_log: int = 40,
        path_lamin_log: float = 1e-2,
        # CV parameters :
        cv: bool = True,
        cv_numerical_method: str = "not specified",
        cv_seed: int = 1,
        cv_one_se: bool = True,
        cv_subsets: int = 5,
        cv__nlam: int = 100,
        cv_lamin: float = 1e-3,
        cv_logscale: bool = True,
        # StabSel parameters :
        stabsel: bool = True,
        stabsel_numerical_method: str = "not specified",
        stabsel_seed: int = None,  # do something here ! for now it can be a bool !
        stabsel_lam: float = -1.0,  # if negative, then it means 'theoretical'
        stabsel_true_lam: bool = True,
        stabsel_method: str = "first",
        stabsel_b: int = 50,
        stabsel_q: int = 10,
        stabsel_percent_ns: float = 0.5,
        stabsel_lamin: float = 1e-2,
        stabsel_threshold: float = 0.7,
        stabsel_threshold_label: float = 0.4,
        # might unneeded here, but needed for visualisation
        # LAMfixed parameters :
        lamfixed: bool = True,
        lamfixed_numerical_method: str = "not specified",
        lamfixed_lam: float = -1.0,  # if negative, then it means 'theoretical'
        lamfixed_true_lam: bool = True,
        # Formulation parameters
        huber: bool = False,
        rho: float = 0.0,
        intercept: bool = True,
) -> classo_problem:
    complete_y = y.to_series()
    complete_y = complete_y[~complete_y.isna()]
    first_cell = complete_y[0]

    # print(sum(complete_y==complete_y[0]), len(complete_y))

    features, pdY = features.align(y.to_series(), join="inner", axis=0)
    missing = pdY.isna()
    training_labels = list(pdY[~missing].index)
    label_missing = list(pdY.index[missing])
    if label_missing:
        print("{} are missing in y ".format(label_missing))
    Y = pdY[~missing].to_numpy()
    X = features.values[~missing, :]

    verfify_binary(Y)
    Y = Y == first_cell
    Y = 2 * Y - 1

    problem = classo_problem(X, Y, C=c, label=list(features.columns))
    problem.formulation.classification = True
    problem.formulation.concomitant = False
    problem.formulation.huber = huber
    # print(rho)
    problem.formulation.rho_classification = rho
    problem.formulation.intercept = intercept
    d = X.shape[1]
    if weights is not None:
        if len(weights) < d:
            problem.formulation.w = np.concatenate([weights, np.ones(d - len(weights))], axis=0)
        else:
            problem.formulation.w = weights[:d]

    problem.model_selection.PATH = path
    if path:
        param = problem.model_selection.PATHparameters
        param.numerical_method = path_numerical_method
        param.n_active = path_n_active
        param.logscale = True
        param.Nlam = path_nlam_log
        param.lamin = path_lamin_log

    problem.model_selection.CV = cv
    if cv:
        param = problem.model_selection.CVparameters
        param.numerical_method = cv_numerical_method
        param.seed = cv_seed
        param.oneSE = cv_one_se
        param.Nsubsets = cv_subsets
        param.lamin = cv_lamin
        param.Nlam = cv__nlam
        param.logscale = cv_logscale

    problem.model_selection.StabSel = stabsel
    if stabsel:
        param = problem.model_selection.StabSelparameters
        param.numerical_method = stabsel_numerical_method
        param.seed = stabsel_seed
        param.true_lam = stabsel_true_lam
        param.method = stabsel_method
        param.B = stabsel_b
        param.q = stabsel_q
        param.percent_nS = stabsel_percent_ns
        param.lamin = stabsel_lamin
        param.threshold = stabsel_threshold
        param.threshold_label = stabsel_threshold_label
        if stabsel_lam > 0.0:
            param.lam = stabsel_lam
        else:
            param.lam = "theoretical"

    problem.model_selection.LAMfixed = lamfixed
    if lamfixed:
        param = problem.model_selection.LAMfixedparameters
        param.numerical_method = lamfixed_numerical_method
        param.true_lam = lamfixed_true_lam
        if lamfixed_lam > 0.0:
            param.lam = lamfixed_lam
        else:
            param.lam = "theoretical"

    problem.solve()

    cy = complete_y.values
    problem.data.complete_y = 2 * (cy == cy[0]) - 1
    problem.data.complete_labels = list(complete_y.index)
    problem.data.training_labels = training_labels

    return problem


def verfify_binary(y):
    lis = []
    for i in y:
        if i not in lis:
            lis.append(i)
    if len(lis) > 2:
        raise ValueError(
            "Metadata column y is supposed to be binary, "
            + "but takes more than 2 different values : "
            + " ; ".join(lis)
        )


def categorical_to_df(col):
    lis = []
    for i in col:
        if i not in lis:
            lis.append(i)
    data = {value: col == value for value in lis}
    return pd.DataFrame(data=data, dtype=np.int8)


def predict(
        features: pd.DataFrame, problem: zarr.hierarchy.Group
) -> prediction_data:
    labels = np.array(problem["data/label"])
    n, d = features.shape[0], len(labels)
    X = np.zeros((n, d))
    for i, label in enumerate(labels):
        if label == "intercept":
            X[:, i] = np.ones(n)
        else:
            X[:, i] = features[label]

    classification = problem["formulation"].attrs["classification"]
    dico_ms = problem["model_selection"].attrs.asdict()
    predictions = prediction_data(
        PATH=problem["model_selection"].attrs["PATH"],
        CV=problem["model_selection"].attrs["CV"],
        StabSel=problem["model_selection"].attrs["StabSel"],
        LAMfixed=problem["model_selection"].attrs["LAMfixed"],
    )

    predictions.sample_labels = list(features.index)

    if predictions.PATH:
        beta_path = np.array(problem["solution/PATH/BETAS"])
        nlam = len(beta_path)
        Y_path = np.zeros((nlam, n))
        for i in range(nlam):
            Y_path[i, :] = X.dot(beta_path[i])
        predictions.YhatPATH = Y_path

    if predictions.CV:
        beta_refit = np.array(problem["solution/CV/refit"])
        Yhat = X.dot(beta_refit)
        predictions.YhatCV = Yhat

    if predictions.StabSel:
        beta_refit = np.array(problem["solution/StabSel/refit"])
        Yhat = X.dot(beta_refit)
        predictions.YhatStabSel = Yhat

    if predictions.LAMfixed:
        beta = np.array(problem["solution/LAMfixed/beta"])
        beta_refit = np.array(problem["solution/LAMfixed/refit"])
        Yhat, Yhatrefit = X.dot(beta), X.dot(beta_refit)
        predictions.YhatLAMfixed = Yhat
        predictions.YhatrefitLAMfixed = Yhatrefit

    return predictions


def _code_columns(df, column_map, norm=1.0, normalization=False):
    # this function takes some dataframe, and transform it so that
    # one can add it to the big dataframe of features
    # ie normalization / transform for categorical
    def _code_col(series):
        type_ = column_map[series.name].type

        if type_ == "numeric":
            vect = series.to_numpy()
            # if vect has a NaN ==> raise error
            if normalization:
                vect = vect - np.mean(vect)
                vect = vect / np.linalg.norm(vect) * norm
            return vect
        elif type_ == "categorical":
            vect = series.to_numpy()
            vect = vect == vect[0]
            to_add = categorical_to_df(series)
            return vect
        else:
            raise ValueError("Unknown type.")

    return df.apply(_code_col, axis=0)


def to_zarr(obj, name, root, first=True):
    '''
    Function for converting a python object to a zarr file , with tree structue.
    '''
    if type(obj) == dict:
        if first:
            zz = root
        else:
            zz = root.create_group(name)

        for key, value in obj.items():
            to_zarr(value, key, zz, first=False)

    elif type(obj) in [np.ndarray, pd.DataFrame]:
        root.create_dataset(name, data=obj, shape=obj.shape)

    elif type(obj) == np.float64:
        root.attrs[name] = float(obj)

    elif type(obj) == np.int64:
        root.attrs[name] = int(obj)

    elif type(obj) == list:
        if name == "tree":
            root.attrs[name] = obj
        else:
            to_zarr(np.array(obj), name, root, first=False)

    elif obj is None or type(obj) in [str, bool, float, int]:
        root.attrs[name] = obj

    else:
        to_zarr(obj.__dict__, name, root, first=first)
