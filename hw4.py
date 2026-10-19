import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('course_lead_scoring_2026.csv')
categorical = list(df.select_dtypes(include=['object', 'string']).columns)
numerical = [c for c in df.columns if c not in categorical and c != 'converted']
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)
features = categorical + numerical

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
y_train = df_train.converted.values
y_val = df_val.converted.values

# Q1
for c in ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print('Q1', c, round(auc, 4))


def train(df_tr, y, C=1.0):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(df_tr[features].to_dict(orient='records'))
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X, y)
    return dv, model


def predict(df_, dv, model):
    X = dv.transform(df_[features].to_dict(orient='records'))
    return model.predict_proba(X)[:, 1]


# Q2
dv, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)
print('Q2 AUC', round(roc_auc_score(y_val, y_pred), 3))

# Q3, Q4
rows = []
for t in np.round(np.arange(0.0, 1.01, 0.01), 2):
    pred = y_pred >= t
    tp = (pred & (y_val == 1)).sum()
    fp = (pred & (y_val == 0)).sum()
    fn = (~pred & (y_val == 1)).sum()
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    rows.append((t, p, r, f1))
m = pd.DataFrame(rows, columns=['t', 'p', 'r', 'f1'])
nz = m[(m.p > 0) | (m.r > 0)]
print('Q3 intersection at', nz.loc[(nz.p - nz.r).abs().idxmin()].round(3).to_dict())
print('Q4 max F1 at', m.loc[m.f1.idxmax()].round(3).to_dict())


# Q5, Q6
def cv(C):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for tr_idx, va_idx in kf.split(df_full_train):
        d_tr, d_va = df_full_train.iloc[tr_idx], df_full_train.iloc[va_idx]
        dv, model = train(d_tr, d_tr.converted.values, C=C)
        scores.append(roc_auc_score(d_va.converted.values, predict(d_va, dv, model)))
    return np.mean(scores), np.std(scores)


mean, std = cv(1.0)
print('Q5 std', round(std, 3), 'mean', round(mean, 3))
for C in [0.000001, 0.001, 1]:
    mean, std = cv(C)
    print('Q6 C =', C, 'mean', round(mean, 3), 'std', round(std, 3))
