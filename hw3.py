import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('course_lead_scoring_2026.csv')
print(df.isnull().sum())

categorical = list(df.select_dtypes(include=['object', 'string']).columns)
numerical = [c for c in df.columns if c not in categorical and c != 'converted']
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)

print('Q1 mode industry:', df.industry.mode()[0])

corr = df[numerical].corr()
print(corr.round(4))
for a, b in [('interaction_count', 'lead_score'), ('number_of_courses_viewed', 'lead_score'),
             ('number_of_courses_viewed', 'interaction_count'), ('annual_income', 'interaction_count')]:
    print('Q2', a, b, round(corr.loc[a, b], 4))

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
y_train = df_train.pop('converted').values
y_val = df_val.pop('converted').values
y_test = df_test.pop('converted').values

for c in categorical:
    print('Q3', c, round(mutual_info_score(df_train[c], y_train), 2))


def train_eval(features, C=1.0):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))
    X_val = dv.transform(df_val[features].to_dict(orient='records'))
    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_val, model.predict(X_val))


features = categorical + numerical

base = train_eval(features)
print('Q4 accuracy:', round(base, 2), base)

for f in features:
    acc = train_eval([x for x in features if x != f])
    print('Q5', f, 'diff =', round(base - acc, 4))

for C in [0.000001, 0.00001, 0.0001, 0.001]:
    print('Q6 C =', C, round(train_eval(features, C=C), 3))