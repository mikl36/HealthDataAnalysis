import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_excel('THL_MyData2016.xls')

# "terveen / normaalin" raja-arvot
terve_bmi_min = 18.5
terve_bmi_max = 25
kolesteroliraja_min = 6
terve_systbp2_min = 130
terve_diastbp2_min = 85
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
df['terve_verenpaine'] = (df['systbp2'] < terve_systbp2_min) & (df['diastbp2'] < terve_diastbp2_min)
terveelliset = df[df['terve_verenpaine']] # df['terve_bmi'] &
ei_terveelliset = df[~(df['terve_verenpaine'])]

# Tarkastetaan terveellisten ja ei-terveellisten osuus sukupuolittain
# ja eri ikäryhmissä, tässä jaottelu bmi, voi tarkentaa myös muut kriteerit
# terveelliset = df[df["bmi"].between(terve_bmi_min, terve_bmi_max)]
# ei_terveelliset = df[~df["bmi"].between(terve_bmi_min, terve_bmi_max)]

# lasketaan määrät, joilla lasketaan suhteelliset osuudet
terveelliset_maara = terveelliset.groupby(["sp", "ikalk"]).size().unstack()
ei_terveelliset_maara = ei_terveelliset.groupby(["sp", "ikalk"]).size().unstack()
kaikki_ei_ja_terveelliset = df.groupby(["sp","ikalk"]).size().unstack()

# suhteelliset osuudet terveelliset ja ei-terveelliset
terveelliset_osuus = terveelliset_maara / kaikki_ei_ja_terveelliset
ei_terveelliset_osuus = ei_terveelliset_maara / kaikki_ei_ja_terveelliset

fig, ax = plt.subplots(figsize=(12, 6))

# Terveelliset miehet / naiset, 1 tai 2
terveelliset_osuus.loc[1].plot.bar(ax=ax, color='green', label="Normaali")
# Ei-terveelliset miehet / naiset, 1 tai 2
ei_terveelliset_osuus.loc[1].plot.bar(ax=ax, color='salmon', bottom=terveelliset_osuus.loc[1], label="Poikkeava")

ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()]) 
# ax.set_title("Normaali- ja yli-/alipainoiset miehet")
ax.set_title("Miesten normaalit ja poikkeavat kolesteroliarvot ikäluokittain") 
# "Normaali- ja yli-/alipainoiset naiset"
ax.set_xlabel("Ikäluokka")
ax.set_ylabel("Suhteellinen osuus")
ax.legend(title="Kolesteroli")

plt.show()

# terveelliset_osuus = terveelliset_osuus.reset_index()

# käsitellään data tulostusta varten
terveelliset_osuus = terveelliset_osuus.T
ei_terveelliset_osuus = ei_terveelliset_osuus.T

terveelliset_osuus.columns = ['Miehet, normaali', 'Naiset, normaali']

ei_terveelliset_osuus.columns = ['Miehet, poikkeava', 'Naiset, poikkeava']

yhdistetty_osuudet = pd.concat([terveelliset_osuus, ei_terveelliset_osuus], axis=1)

# trendiviivakaaviot terveellisten ja ei-terveellisten osuuksista ikäluokittain

plt.figure()
sns.lineplot(data=yhdistetty_osuudet * 100, markers=True, palette="flare")
plt.xlabel("Ikäluokka")
plt.ylabel("Osuus (%)")
plt.title("Normaali- ja korkea-arvoiset verenpaineet ikäluokittain")
# plt.title("Normaali- ja yli-/alipainoisten osuudet ikäluokittain")
plt.legend()

plt.show()


# plt.figure(figsize=(12, 6))
# sns.barplot(yhdistetty_osuudet)

# sns.barplot(x='ikalk', y= 'bmi', hue= 'sp', data=df)
# plt.show()





