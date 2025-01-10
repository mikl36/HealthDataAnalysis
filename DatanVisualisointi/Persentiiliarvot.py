import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, spearmanr

df = pd.read_excel('THL_MyData2016.xls')

# Valitse ikäryhmä(t) "30-39", "40-49", "50-59", "60-69", "70-79", "80-110"
ikaryhmat = ["30-39"]

# Suodatus ikäryhmällä
suodatettu_df = df[df["ikalk"].isin(ikaryhmat)]

# sukupuolen valinta (1 = miehet, 2 = naiset)
sukupuoli = 2

# Suodatus sukupuolella
suodatettu_df = suodatettu_df[suodatettu_df["sp"] == sukupuoli]

# valita sarakkeet, jotka kaaviolle: "bmi", "systbp2", "diastbp2", "kol", "kol_hdl"
sarakkeet = ["bmi", "systbp2", "diastbp2", "kol", "kol_hdl"]

# viivakaavio, normaali
# plt.figure(figsize=(12, 5))
# for ikaryhma in ikaryhmat: # suodatus sukupuolella ja ikaryhmalla
#    suodatettu_df = df[(df["ikalk"] == ikaryhma) & (df["sp"] == sukupuoli)]
#    for s in sarakkeet:
#        plt.plot(suodatettu_df["Fx"], suodatettu_df[s], marker='o', label=f"{ikaryhma}, {s}")
        
    
# viivakaavio logaritmisella asteikolla, arvoerot
plt.figure(figsize=(10, 6))
for ikaryhma in ikaryhmat: # suodatus sukupuolella ja ikaryhmalla
#   for sukupuoli in [1, 2]: # jos molemmat sukupuolet, sisentää alemmat
    suodatettu_df = df[(df["ikalk"] == ikaryhma) & (df["sp"] == sukupuoli)]
    for s in sarakkeet:
        plt.semilogy(suodatettu_df["Fx"], suodatettu_df[s], marker='o', label=f"{ikaryhma}, {s}")
#            plt.semilogy(suodatettu_df["Fx"], suodatettu_df[s], marker='o', # jos molemmat sukupuolet
#            label=f"{ikaryhma}, {'Miehet' if sukupuoli == 1 else 'Naiset'}, {s}")

plt.xlabel("Persentiili Fx")
plt.ylabel("Arvo")
# plt.title(f"Terveysindikaattorit ikäryhmittäin ({'Miehet' if sukupuoli == 1 else 'Naiset'})")
# plt.title("Terveysindikaattorit ikäryhmittäin ja sukupuolittain")
plt.title(f"Terveysindikaattorit ikäryhmässä {ikaryhmat} ({'Miehet' if sukupuoli == 1 else 'Naiset'})")
plt.legend()
plt.grid(True)
plt.show()

