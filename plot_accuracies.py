X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=(SEED+1))
rashomon_sets = {}
rashomon_sets_acc_lower_bounds = {}
accuracies = {}
for model_class in MODELS:
    rashomon_sets[model_class.__name__] = []
    accuracies[model_class.__name__] = []
    for kwargs in rashomon_sets_params[model_class.__name__]:
        if model_class.__name__ == 'SVMClassifier':
            model = model_class(probability=True, **kwargs)
        else:
            model = model_class(**kwargs)
        model.fit(X_train, y_train)
        acc = np.mean(model.predict(X_test) == np.array(y_test))
        accuracies[model_class.__name__].append(acc)
        rashomon_sets[model_class.__name__].append(model)
    rashomon_sets_acc_lower_bounds[model_class.__name__] = min(accuracies[model_class.__name__])