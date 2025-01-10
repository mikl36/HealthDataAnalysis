import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr


df = pd.read_excel('THL_MyData2016.xls')

# tarkastetaan aineisto: tyhjät arvot, poikkeamat arvoissa 
# (arvot järkeviä, arvojen sijainnit järkeviä jne.), tunnusluvut, 
# tarkastetaan, onko NaN-arvoja, ei ole
nan_arvot = df.isna().sum()

# tarkastetaan aineisto poikkeamien varalta, 'sp', 
# 'bmi', 'systbp2', 'diastbp2', 'kol', 'kol_hdl', 'Fx_plus', 'Fx'
# tarkasteluarvo = 'bmi';
# print (df[tarkasteluarvo].unique())
# print (df[tarkasteluarvo].nlargest(15))
# print (df[tarkasteluarvo].nsmallest(15))

# Spectral, flare, Set2, pastel
sns.scatterplot(data=df, x="kol", y="kol_hdl", hue="ikalk", palette="Spectral")
plt.show()

# tunnusluvut, eivät täysin toimiva tässä, koska persentiilit
# desc = df.describe()
# df.info()

# 50 persentiili tai muiden persentiilien tarkastelu
persentiili = df.groupby(["sp", "ikalk"]).quantile(0.5)

# print(persentiili)


# tutkittavat sarakkeet (esimerkiksi "bmi")
sarakkeet = ["bmi", "systbp2", "diastbp2", "kol", "kol_hdl"]

# pylväskaavio, ikalk tai sp
# plt.figure(figsize=(10, 6))
# for i, s in enumerate(sarakkeet):
#     plt.subplot(3, 5, i + 1)
#     plt.bar(persentiili.index.get_level_values("ikalk"), persentiili[s])
#     plt.title(s)
#     plt.xlabel("Ikäluokat")
#     plt.ylabel("Keskiarvo")
#     plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()

# pylväskaavio sukupuolittain ja ikäluokittain
# plt.figure(figsize=(10, 6))
# for i, s in enumerate(sarakkeet):
#     plt.subplot(3, 5, i + 1) # rivit, sarakkeet, sijainti
#     for gender in persentiili.index.get_level_values("sp").unique():
#        sp_persentiili = persentiili.loc[gender]
#        plt.bar(sp_persentiili.index.get_level_values("ikalk"), sp_persentiili[s], label=gender, alpha=0.7)
#    plt.title(s)
#    plt.xlabel("Ikäluokat")
#    plt.ylabel("Mediaani")
#    plt.xticks(rotation=45)
#    plt.legend(loc='lower right')

#plt.tight_layout()
#plt.show()



