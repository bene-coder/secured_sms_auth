import pickle, pandas as pd

with open('posttest_result/trained_voting_model.pkl', 'rb') as f:
    d = pickle.load(f)

cols = d['feature_columns']
print(len(cols), cols)

imp = pd.DataFrame({
    'feature': cols,
    'rf_importance_%': d['random_forest'].feature_importances_ * 100,
    'xgb_importance_%': d['xgboost'].feature_importances_ * 100,
    'lr_abs_coef': abs(d['logistic_regression'].coef_[0]),
})
print(imp.sort_values('rf_importance_%', ascending=False).round(2).to_string(index=False))
print('threshold:', d.get('decision_threshold'))