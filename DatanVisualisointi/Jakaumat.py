import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr


df = pd.read_excel('THL_MyData2016.xls')

# Tarkastetaan jakaumat yleisesti: bmi, verenpaineet, kolesterolit
# voidaan jaotella vielä sukupuolittain ja/tai ikäluokittain
df.plot.hist(y=["kol_hdl", "kol"], bins=16, alpha=0.5, figsize=(12, 6))
plt.title("HDL- ja kokonaiskolesterolin jakautuminen")
plt.xlabel("mmol/l")
plt.ylabel("Frekvenssi")
plt.legend(["HDL-kolesteroli", "Kokonaiskolesteroli"])
plt.show()

# Yksittäisen tai useamman yksittäisen histogrammin tulostus
# Määritetään tulostettavat arvot ja otsikot
# tulostettavat_arvot = {
#    "kol_hdl": ("mmol/l", "HDL-kolesterolin jakautuminen"),
#    "kol": ("mmol/l", "Kolesterolin jakautuminen"),
#    "systbp2": ("mmHg", "Systolisen verenpaineen jakautuminen"),
#    "diastbp2": ("mmHg", "Diastolisen verenpaineen jakautuminen"),
#    "bmi": ("kg/m^2", "Painoindeksin, BMI, jakautuminen"),
#}

# Tulostetaan tulostettavat arvot
#for nimi, (xlabel, otsikko) in tulostettavat_arvot.items():
    # Histogrammi
#    df.plot.hist(y=[nimi], bins=15, alpha=0.5, figsize=(12, 6), color="green")

#    plt.title(otsikko)
#    plt.xlabel(xlabel)
#    plt.ylabel("Frekvenssi")
#    plt.legend([nimi])

#    plt.show()
    
# histogrammi verenpaine
df.plot.hist(y=["systbp2", "diastbp2"], bins=10, alpha=0.5, figsize=(10, 6))
plt.title("Systolisen ja diastolisen verenpaineen jakautuminen")
plt.xlabel("mmHg")
plt.ylabel("Frekvenssi")
plt.legend(["Systolinen", "Diastolinen"])
plt.show()    

# tarkastetallaan miesten ja naisten välisiä eroja
# suodatetaan miesten, 1, arvot
miehet = df[df['sp'] == 1]

# suodatetaan naisten, 2, arvot
naiset = df[df['sp'] == 2]

# histogrammit
plt.hist(miehet['bmi'], bins=16, alpha=0.5, color='blue', label='Miehet')
plt.hist(naiset['bmi'], bins=16, alpha=0.4, color='red', label='Naiset')

# naisten ja miesten ka-viivat, 50 persentiilit, tai jokin muu raja 
plt.axvline(miehet['bmi'].quantile(0.5), color='blue', linestyle='--', linewidth=2, label='Miesten keskiarvo')
plt.axvline(naiset['bmi'].quantile(0.5), color='red', linestyle='--', linewidth=2, label='Naisten keskiarvo')

plt.title("Painoindeksin, BMI, jakautuminen sukupuolittain")
plt.xlabel("kg/m^2")
plt.ylabel("Frekvenssi")
plt.legend()
plt.show()

# Suodata miesten arvot vertailevissa ikäluokissa
miehet_30_39 = df[(df['sp'] == 1) & (df['ikalk'] == '30-39')]
miehet_70_79 = df[(df['sp'] == 1) & (df['ikalk'] == '70-79')]

# Suodata naisten arvot vertailevissa ikäluokissa
naiset_30_39 = df[(df['sp'] == 2) & (df['ikalk'] == '30-39')]
naiset_70_79 = df[(df['sp'] == 2) & (df['ikalk'] == '70-79')]

# histogrammit vertailu esim. sukupuolittain ja/tai ikäluokittain
plt.hist(miehet_30_39['bmi'], bins=16, alpha=0.5, color='blue', label='Miehet 30-39')
plt.hist(miehet_70_79['bmi'], bins=16, alpha=0.4, color='cyan', label='Miehet 70-79')
# plt.hist(naiset_30_39['bmi'], bins=16, alpha=0.4, color='red', label='Naiset 30-39')
# plt.hist(naiset_70_79['bmi'], bins=16, alpha=0.5, color='blue', label='Naiset 70-79')

plt.title("Painoindeksin, BMI, jakautuminen miehissä ikäluokittain 30-39 ja 70-79")
plt.xlabel("kg/m^2")
plt.ylabel("Frekvenssi")
plt.legend()
plt.show()

# kol, kol_hdl, bmi, diastbp2, systbp2
# voisi suodattaa esim. vanhemmat ikäluokat vielä pois
sns.histplot(x="systbp2", data = df, hue="sp")
plt.show()

sns.histplot(x="systbp2", data = miehet, hue="ikalk")
plt.show()

sns.histplot(x="bmi", data = naiset, hue="ikalk")
plt.show()

