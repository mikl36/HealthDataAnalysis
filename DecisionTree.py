import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix

df = pd.read_excel('THL_MyData2016.xls')

def laske_terveystilanne(df):
# "terveen / normaalin" raja-arvot, höllennetyt rajat
    terve_bmi_min = 18.5
    terve_bmi_max = 30 # . lievä lihavuus
    kolesteroliraja_min = 7 # lievästi kohonnut
    terve_systbp2_muut_min = 140 # tyydyttävä
    terve_systbp2_vanhin_ikalk_min = 150 # vanhin ikäluokka 80 alkaen
    terve_diastbp2_min = 90 # tyydyttävä
    terve_hdl_min_naiset = 1.2
    terve_hdl_min_miehet = 1.0

# tervekriteerit true tai false aineistolla
    df['terve_hdlkolesteroli'] = np.where(
        (df['sp'] == 1) & (df['kol_hdl'] >= terve_hdl_min_miehet) | 
        (df['sp'] == 2) & (df['kol_hdl'] >= terve_hdl_min_naiset),
        True, False
        )
    df['terve_bmi'] = df['bmi'].between(terve_bmi_min, terve_bmi_max)
    df['terve_kol'] = df['kol'] < kolesteroliraja_min
    df['terve_verenpaine'] = np.where(
        ((df['ikalk'] == '80-110') & (df['systbp2'] < terve_systbp2_vanhin_ikalk_min) 
         & (df['diastbp2'] < terve_diastbp2_min)) |
        ((df['ikalk'] != '80-110') & (df['systbp2'] < terve_systbp2_muut_min) 
         & (df['diastbp2'] < terve_diastbp2_min)),
        True, False
        )
    terveelliset = df[(df['terve_verenpaine']) & (df['terve_kol']) & (df['terve_bmi']) & df['terve_hdlkolesteroli']] 
    # lisätään sarake 'terveystilanne' sen mukaan onko se normaali 1, poikkeava 0
    df['terveystilanne'] = np.where(df.index.isin(terveelliset.index), 1, 0)
    return df

# ei_terveelliset = df[~((df['terve_verenpaine']) & (df['terve_kol']) & (df['terve_bmi']) & df['terve_hdlkolesteroli'])]

df = laske_terveystilanne(df)

# X and y data for the model
X = df.iloc[:, [0,1,2,3,4,5,6]]
y = df.iloc[:, [-1]]

Xorg = X;

# making dummy-variables, drop first 30-39 and 1, men
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['ikalk','sp'])], remainder='passthrough')
X = ct.fit_transform(X) 

y = y.values.ravel()

# Splitting the dataset into the Training set and Test set 20-80
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
# Training the Decision Tree Classification model
model = tree.DecisionTreeClassifier(max_depth= 10, criterion='gini') # ei parane enää
model.fit(X_train, y_train)

mfi = model.feature_importances_

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred_pros = model.predict_proba(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)

print (f'{cm}')
print (f'Mallin ulkoinen tarkkuus: {ac*100:.02f} %')
print (f'precision_score: {ps:.02f}')
print (f'recall_score: {rc:.02f}')

# sns.heatmap(cm, annot=True, fmt='g')
# plt.show()

tn, fp, fn, tp = cm.ravel()
ax = plt.axes()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
plt.title("Confusion Matrix, DT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# to check the original values, what went wrong
wrong_values = pd.DataFrame(data=X_test)
wrong_values['pred'] = y_pred
wrong_values['actual'] = y_test

# false positive or false negative index
fn_indeksi = np.where((y_test == 1) & (y_pred == 0))[0]
fp_indeksi = np.where((y_test == 0) & (y_pred == 1))[0]

fn_fp_values = wrong_values.loc[np.concatenate([fn_indeksi, fp_indeksi])]

# Create dot file for graphviz visualization
dot_data = export_graphviz(
            model,
            out_file =  None,
            feature_names = ['40-49', '50-59', '60-69', '70-79', '80-110', '2',
             'bmi', 'systbp2', 'diastbp2', 'kol', 'kol_hdl'],
            class_names = ['Normaalit_arvot', 'Poikkeavat_arvot'],
            filled = True,
            rounded = True)

# graph = graphviz.Source(dot_data)
# graph.render(filename = 'THL_MyData2016', format = 'png')

# Predicting based on new data
df_new = pd.read_csv('testiaineisto.csv')
df_new_org = df_new
df_new = ct.transform(df_new)

y_new = model.predict(df_new)
y_new_proba = model.predict_proba(df_new)

# for i in range(5):
#    print (f'{df_new_org.iloc[i]}\nTerveystilanne: {y_new[i]} ({y_new_proba[i][1]:.02f})')

# tervekriteerit true tai false aineistolla
df_new_org = laske_terveystilanne(df_new_org)

terveystilanne_new = df_new_org['terveystilanne']

# metrics
cm_new = confusion_matrix(y_new, terveystilanne_new)
tn, fp, fn, tp = cm.ravel()

ax = plt.axes()
sns.heatmap(cm_new, annot=True, fmt='g', ax=ax)
plt.title("Confusion Matrix, New Values, DT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

ac_new = accuracy_score(terveystilanne_new, y_new)
print(f'Ennustuksen ulkoinen tarkkuus: {ac_new*100:.02f} %')




