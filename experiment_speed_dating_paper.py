from PyALE import ale


def make_an_experiment(model, X):
    ## 2D - continuous
    ale_eff = ale(X=X, model=model, feature=["intelligence", "intelligence_partner"], grid_size=100)
    ale_eff = ale(X=X, model=model, feature=["ambition", "ambition_partner"], grid_size=100)
