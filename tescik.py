from pytorch_tabr import TabRClassifier

clf = TabRClassifier(
    cat_indices=cat_idxs,
    cat_cardinalities=cat_dims,
    type_embeddings="one-hot",
    device_name="cpu",
    optimizer_params={"lr": 2e-4},
    d_main=96,
    context_size=96,
    # selection_function_name="sparsemax",
    # context_dropout=0.5,
    # context_sample_size=2000,
    # num_embeddings={"type": "PLREmbeddings", "n_frequencies": 32, "frequency_scale": 32, "d_embedding": 32, "lite": False},
)

clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_valid, y_valid)], max_epochs=2, batch_size=2048)

preds = clf.predict_proba(X_test)
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)

preds_valid = clf.predict_proba(X_valid)
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)