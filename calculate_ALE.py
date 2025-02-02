from PyALE import ale


def getALE(model, X):
    ## 1D - continuous - with 95% CI
    ale_eff = ale(
        X=X, model=model, feature=["attractive_partner"], grid_size=50, include_CI=True, C=0.95
    )

