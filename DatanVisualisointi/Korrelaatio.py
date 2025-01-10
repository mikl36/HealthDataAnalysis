import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, norm, pearsonr, spearmanr, ttest_ind, f_oneway, goodness_of_fit, levene
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics import r2_score

df = pd.read_excel('THL_MyData2016.xls')

# sukupuolien suodatus
df_miehet = df[df["sp"] == 1]
df_naiset = df[df["sp"] == 2]

sarakkeet = ['bmi', 'systbp2', 'diastbp2', 'kol', 'kol_hdl']

# testataan normaalijakauma ja varianssit, voisi/tulisi testata myös ikäryhmien
for s in sarakkeet:
   print(f"Sarake: {s}")
    
    # normaalijakauman testaus
   gf = goodness_of_fit(norm, df_miehet[s])
   print(f"Goodness of Fit (normaalijakauma) p-arvo: {gf.pvalue}")
    
    # Levene-testi
#    levene_result = levene(df_miehet[s], df_naiset[s])
#    print(f"Levene-testin tulos: {levene_result}\n")

# bmi, systbp2, diastbp2, kol_hdl: ei normaalijakautunut
# kol niukasti

# variansseissa samankaltaisuutta ainoastaan diastbp2 (niukasti) ja kol
# (Welchin t-testi, mutta ongelmana normaalijakaumat) -> Mann-Whitney test

# korrelaatiomatriksi, kaikki
#correlation_matrix = df[sarakkeet].corr()
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
#plt.title('Korrelaatio BMI:n, verenpaineen ja kolesterolin välillä')
# plt.show()

# p-arvot, pearson, tässä koko aineistolla
#for s1 in sarakkeet:
#    for s2 in sarakkeet:
#        if s1 != s2:
#            corr, p_value = pearsonr(df[s1], df[s2])
#            r2 = corr**2
#            print(f"Korrelaatio ({s1}, {s2}): {corr:.2f}, p-arvo: {p_value:.4f}")
#            print(f"R2-luku ({s1} vs. {s2}): {r2:.4f}")
            
# korrelaatiomatriksi, miehet / naiset
# correlation_matrix = df_miehet[['bmi', 'systbp2', 'diastbp2', 'kol', 'kol_hdl']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
# plt.title('Miesten korrelaatio BMI:n, verenpaineen ja kolesterolin välillä')
# plt.show()

# regressiosuora vs scatter ikäluokittain bmi, verenpaine
# sns.regplot(x='diastbp2', y='systbp2', data=df)
# plt.title('Diastolinen vs. Systolinen verenpaine')
# plt.xlabel('Diastolinen, mmHg')
# plt.ylabel('Systolinen, mmHg')
# plt.show()

# sns.scatterplot(x='diastbp2', y='systbp2',hue="ikalk", data=df)
# plt.title('Diastolinen vs. Systolinen verenpaine')
# plt.xlabel('Diastolinen, mmHg')
# plt.ylabel('Systolinen, mmHg')
# plt.show()

# onko naisten ja miesten BMI-keskiarvot (esim.) tilastollisesti merkitsevästi erilaisia
# nollahypoteesi: ryhmien keskiarvot ovat samat

# tehdään Mann-Whitney testillä kaikki, vaikka osan voi tehdä t-testillä (t-testillä samat tulokset)
# for s in sarakkeet:
#    u_statistic, p_value = mannwhitneyu(df_miehet[s], df_naiset[s])
#    print(f"Indikaattorin {s} U-arvo: {u_statistic:.4f}, p-arvo: {p_value:.4f}")
#    if p_value < 0.05:
#        print(f"Sukupuolten {s} keskiarvot ovat tilastollisesti merkitsevästi erilaisia.")
#    else:
#       print(f"Sukupuolten {s} keskiarvot eivät ole tilastollisesti merkitsevästi erilaisia.")


# ANOVA ikäryhmien ja sukupuolen mukaan, nollahypoteesi: ryhmien keskiarvot samat
# tehdään Kruskal-Wallis, koska aineiston normaalijakautuneisuudessa ja variansseissa
# ANOVALLA BMI ei täyty, ANOVA voi olla parempi tähän, jos tarkastetaan aineisto
# ehdot eivät täyty suuressa osassa (ei testattu ikäryhmien variansseja)
for s in sarakkeet:
   result1 = kruskal(
#        df_naiset[df_naiset['ikalk'] == '30-39'][s],
#        df_naiset[df_naiset['ikalk'] == '40-49'][s],
#        df_naiset[df_naiset['ikalk'] == '50-59'][s],
        df_naiset[df_naiset['ikalk'] == '60-69'][s],
        df_naiset[df_naiset['ikalk'] == '70-79'][s],
        df_naiset[df_naiset['ikalk'] == '80-110'][s]
   )
    
   result2 = kruskal(
#                    df_miehet[df_miehet['ikalk'] == '30-39'][s],
#                    df_miehet[df_miehet['ikalk'] == '40-49'][s],
#                    df_miehet[df_miehet['ikalk'] == '50-59'][s],
                    df_miehet[df_miehet['ikalk'] == '60-69'][s],
                    df_miehet[df_miehet['ikalk'] == '70-79'][s],
                    df_miehet[df_miehet['ikalk'] == '80-110'][s]                         
                    )

# ainakin yhden poikkeaa

#   if result1.pvalue < 0.05:
#        print(f"Indikaattorin {s} keskiarvot ovat tilastollisesti merkitsevästi erilaisia.")
#   else:
#        print(f"Indikaattorin {s} keskiarvot eivät ole tilastollisesti merkitsevästi erilaisia.")
      
