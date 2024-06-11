# %% [markdown]
# # Packages necessary
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from scipy.interpolate import interp1d
from datetime import timedelta
from scipy.stats import t # type: ignore
from scipy.stats import gmean
from scipy.optimize import minimize
from datetime import datetime
from statistics import mean

# %% [markdown]
# # Notes sur le modèle CIR
# 
# - Nous remarquons que cela est très similaire au modèle de Vasicek, avec l'ajout unique du terme $\sqrt(r_t)$.
# - Ce terme tendra vers zéro lorsque le taux court approche de zéro, éliminant efficacement la volatilité lorsque le taux court diminue.
# - L'ajout de ce terme forcera $r_t$ à rester non négatif dans le modèle CIR, contrairement au modèle de Vasicek.
# - Cependant, un inconvénient majeur est que l'ajout du terme limitant la volatilité rend le modèle CIR, contrairement au modèle de Vasicek, non gaussien.(En d'autres termes, les variations du taux d'intérêt ne suivent plus une distribution gaussienne, mais une autre distribution plus complexe.)
# 

# %%
# def cir_simulation(r0, a, b, sigma, T, N, num_paths, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
    
#     dt = T / N
#     rates = np.zeros((N+1, num_paths))
#     rates[0] = r0
    
#     for i in range(num_paths):
#         for t in range(1, N+1):
#             gamma = np.sqrt(a**2 + 2*sigma**2)
#             dr = a * (b - rates[t-1, i]) * dt + sigma * np.sqrt(rates[t-1, i]) * np.random.normal(0, np.sqrt(dt))
#             rates[t, i] = max(rates[t-1, i] + dr, 0)  # Ensure rates remain non-negative
#     return rates


# %%
# # Paramètres
# r0 = 0.05  # Taux court initial
# a = 0.1    # Vitesse d'ajustement
# b = 0.07   # Moyenne à long terme
# sigma = 0.01  # Volatilité
# T = 10.0    # Horizon temporel
# N = int(52*T)    # Nombre de pas de discrétisation
# num_paths = 1  # Nombre de chemins à simuler


# # Simulation
# simulated_rates=cir_simulation(r0, a, b, sigma, T, N, num_paths, seed=14)

# # # Visualisation
# # plt.figure(figsize=(10, 6))
# # plt.plot(np.linspace(0, T, N+1), simulated_rates)
# # plt.title('Simulation du modèle de Vasicek')
# # plt.xlabel('Temps')
# # plt.ylabel('Taux court')
# # plt.grid(True)
# # plt.show()

# %%
df=pd.read_excel('df.xlsx')
df_c=df.copy()
r_t = df_c['1 J']
r_t


# %% [markdown]
# # Optimization

# %%
# Convertir la colonne des dates en datetime

df_c['Date de référence'] = pd.to_datetime(df_c['Date de référence'])
df_c = df_c.sort_values(by='Date de référence')

# Calculer la différence entre les dates en années
df_c['dt'] = df_c['Date de référence'].diff().dt.total_seconds() / (365.25 * 24 * 3600)

# Calculer la moyenne des valeurs de la colonne "dt" (en excluant la première valeur)
moyenne_dt = df_c['dt'][1:].mean()

# Remplacer la première valeur par la moyenne
df_c['dt'].iloc[0] = moyenne_dt
df_c['dt'] = df_c['dt'].astype(float)
delta=df_c['dt']
try:
    df_c = df_c.drop(columns=['difference'])
except KeyError:
    pass
# Afficher le DataFrame
print(df_c)


# %%
delta

# %%
import statsmodels.api as sm       

delta=df_c['dt']
Y_t = np.roll(r_t,1) / np.sqrt(r_t*delta )       
X_t = 1 / np.sqrt(r_t*delta)       
Z_t = np.sqrt(r_t)/np.sqrt(delta)      


# Add constant term to independent variables       
X = np.column_stack((X_t, Z_t))       


# %%

# Fit regression model       
model = sm.OLS(Y_t, X).fit()       

# Get model summary       
print(model.summary())       
      
# Calculate MSE       
sigma3 = np.sqrt(model.mse_resid)       
print("sigma = ", sigma3)       

a3 = (1-model.params[1])        
print("a     = ", a3)       

b3 = model.params[0]/(a3)       
print("b     = ", b3)       


# %%
maturities = np.array(
    [ 1/365.25  , 7/365.25, 1/12, 2/12 , 3/12, 6/12, 9/12, 1 , 2 , 3 , 4 , 5 , 6 , 
  7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 
  19 , 20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30]
)

# %%
def A(x, a, b, sigma):
    gamma = np.sqrt(a**2 + 2 * sigma**2)
    numerator = 2 * gamma * np.exp((a + gamma) * (x) / 2)
    denominator = (a + gamma) * (np.exp(gamma * (x)) - 1) + 2 * gamma
    return (numerator / denominator) ** (2 * a * b / sigma**2)

def B(x, a, sigma):
    gamma = np.sqrt(a**2 + 2 * sigma**2)
    numerator = 2 * (np.exp(gamma * (x)) - 1)
    denominator = (a + gamma) * (np.exp(gamma * (x)) - 1) + 2 * gamma
    d=numerator / denominator
    return d

def P(x, rt, a, b, sigma):
    return A(x, a, b, sigma) * np.exp(-B(x, a, sigma) * rt)

# %%
def P_vasi(x, rt, a, b, sigma):
    B_x = (1 - np.exp(-a * x)) / a
    A_x = np.exp(((B_x - x) * (a**2 * b - (sigma**2)/2)) / (a**2) - ((sigma**2 * B_x**2) / (4 * a)))
    return A_x * np.exp(-B_x * rt)

# %%
def R(x, r, a, b, sigma):
    return np.log(P(x, r, a, b, sigma))/(-x) if x!=0 else  r # here x is T-t !!!


def R_vasi(x, r, a, b, sigma):
    return np.log(P_vasi(x, r, a, b, sigma))/(-x) if x!=0 else  r # here x is T-t !!!

# %%
def objective(params:list):
    """ 
    T
    """ 
    a, b, sigma = params
    error = 0
    for row in df_c.values:
        for i in range(1, len(row)):  # Commencer à partir de la deuxième colonne
            t = row[0]
            T = maturities[i-1]  # Indexation de 0 pour la première colonne, donc i-1 pour maturities
            P = row[i]
            YY_YT = R(T, row[1], a, b, sigma) # les Y estimés
            error += (P - YY_YT)**2
    return error
           
# # Valeurs initiales pour a, b, sigma
# initial_guess = [0.1, 0.1, 0.01]
# # --------------------------Optimization--------------------------------
# # Minimiser la fonction objective
# result = minimize(objective, initial_guess, method='Nelder-Mead')

# # Les valeurs optimales de a, b, sigmassssssssssssaqwwwwwwwwwssssza
# a_opt, b_opt, sigma_opt = result.x

# print("Valeurs optimales de a, b, sigma:", a_opt, b_opt, sigma_opt)

# # Les valeurs optimales de a, b, sigma
# a_opt, b_opt, sigma_opt = result.x

# print("Valeurs optimales de a, b, sigma:", a_opt, b_opt, sigma_opt)

# %%
sigma3

# %%
# a, b, sigma= 0.005487472530570595, 0.5571411225041425, 0.08329893264240101 # nelder Mead
# a, b, sigma=a_opt, b_opt, 0.02
# ML
# a = 0.1
# b = 0.05
# sigma = 0.02

# Estimation de alpha: 0.011378255165944127
# Estimation de mu: 0.027605692944953873
# Estimation de sigma: 0.007090485127573163
a= 0.1137805121585627
b =0.05011965977532673
sigma= 0.019999641509027735


# a=0.011378255165944127
# b= 0.027605692944953873
# # sigma=0.5
# sigma =  0.007103662246569265
# a     =  0.011538576605948814
# b     =  0.027455280630213262


# %%
dd_est = df_c.copy() 
try:  
    dd_est = dd_est.drop(columns=['dt'])
except:
    pass
matrixx = []            

for row in dd_est.values:
    for i in range(1, len(row)):  # Commencer à partir de la deuxième colonne 
        T = maturities[i-1]  # Indexation de 0 pour la première colonne, donc i-1 pour maturities
        YY_YT = R(T, row[1], a, b, sigma)
        row[i] = YY_YT
    matrixx.append(list(row))
data_Modeled_CIR=pd.DataFrame(matrixx,columns=dd_est.columns)    

# %%
# Date spécifique pour laquelle tracer les courbes
date_specifique = "2023-10-16"
daf=df_c
donnees_modeled_specifique=data_Modeled_CIR[data_Modeled_CIR["Date de référence"]==date_specifique]
donnees_date_specifique = df[df["Date de référence"] == date_specifique]


# %%
taux_observés = donnees_date_specifique.iloc[0, 1:].values.astype(float)

taux_modélisés = donnees_modeled_specifique.iloc[0, 1:].values.astype(float)

# %%
# Tracer les courbes des taux zéro-coupon observés et modélisés
plt.plot(maturities, taux_observés, label='Taux zéro-coupon observés')
plt.plot(maturities, taux_modélisés, label='Taux zéro-coupon modélisés')
plt.xlabel('maturités (en années)')
plt.ylabel('Taux')
plt.title('Courbe des taux zéro-coupon observés et modélisés pour le ' + date_specifique)
plt.legend()
plt.grid(True)
plt.show()


# %%
donnee=data_Modeled_CIR.copy()
def calculate_aic(y_obs, y_pred, num_params):
    resid = y_obs - y_pred
    rss = np.sum(resid**2)
    aic = 2 * num_params - 2 * np.log(rss)
    return aic

def calculate_bic(y_obs, y_pred, num_params, num_obs):
    resid = y_obs - y_pred
    rss = np.sum(resid**2)
    bic = num_obs * np.log(rss / num_obs) + num_params * np.log(num_obs)
    return bic

# Suppose data.iloc[:,1] est les données observées et data_Modeled_CIR[0] sont les données modélisées
y_obs = df['1 J'] # Les taux d'intérêt observés
y_pred = donnee['1 J']  # Les taux d'intérêt modélisés
num_params = 3  # Nombre de paramètres dans le modèle (a, b, sigma)
num_obs = len(y_obs)  # Nombre total d'observations

# Calcul de l'AIC et du BIC
aic = calculate_aic(y_obs, y_pred, num_params)
bic = calculate_bic(y_obs, y_pred, num_params, num_obs)

print("AIC:", aic)
print("BIC:", bic)

# %% [markdown]
# Pour le modele de CIR ona : 
# $$ AIC: 6.892506486240845 \quad
# BIC: -38753.191465424316 $$
# pour Le modele de Vasicek :
# 
# $$ AIC: 38.67852469897918 \quad
# BIC: -108549.2272198336 $$
# 
# Donc le meilleur modele est celui de *Vasicek*

# %%
maturities

# %%
data=pd.read_excel("df.xlsx")
data.columns=['Date de référence']+ list(maturities)
# data.to_excel("donnees_modifiees.xlsx", index=False)
# data[data["Date de référence"]=="2023-03-03"]
# data.iloc[0]

# %%


def find_nearest_date(date, data):
    nearest_date = data.iloc[(data['Date de référence'] - date).abs().idxmin()]
    return nearest_date.iloc[1:]

def Valorisation_BDT(tx, N, r, d_val, d_ech):
        date_ver_c = []
        date_ver_c.append(d_ech)
        x = d_ech
        while x - timedelta(days=365.25) > d_val:
            date_ver_c.append(x)
            x = x - timedelta(days=365.25)
        date_ver_c.sort()
        
        mat = [(date - d_val).days / 365.25 for date in date_ver_c]
        
        zc = interp1d(maturities,r,kind='linear',fill_value='extrapolate')
        valeur = 0
        for i in range(len(mat)):
            t=mat[i]
            # valeur += (N * tx / (1 + int_r(r,t) if t<30 else int_r(r,30) ) ** t)
            valeur += (N * tx / (1 + zc(t)) ** t)
            
        # valeur += N / (1 + int_r(r,mat[-1])) ** mat[-1]
        valeur += N / (1 + zc(mat[-1])) ** mat[-1]

                
        return valeur
# d_val = datetime.strptime("04/03/2023", "%d/%m/%Y")
# d_ech = datetime.strptime("07/06/2026", "%d/%m/%Y")
# g=Valorisation_BDT( 0.062,210000000, data.iloc[0,1:], d_val, d_ech)


# %%
def find_nearest_date(date, data):
    nearest_date = data.iloc[(data['Date de référence'] - date).abs().idxmin()]
    return nearest_date.iloc[1:]

# %%
from datetime import datetime
# Convertir la chaîne de caractères en objet datetime
date = datetime.strptime("2023-03-03", "%Y-%m-%d")
# Utiliser la fonction find_nearest_date avec l'objet datetime
nearest_r = find_nearest_date(date, data)
date

# %%
# Chargement des données depuis le fichier Excel
data_BDT = pd.read_excel('BDT.xlsx')
data_Valorisation_Date=data_BDT.copy()

# %%
# Conversion des colonnes 'dt/val' et 'dt/ech' en objets datetime
data_Valorisation_Date['dt/val'] = data_Valorisation_Date['dt/val'].apply(lambda x: x.to_pydatetime())
data_Valorisation_Date['dt/ech'] = data_Valorisation_Date['dt/ech'].apply(lambda x: x.to_pydatetime())

def find_nearest_date(date, data):
    nearest_date = data.iloc[(data['Date de référence'] - date).abs().idxmin()]
    return nearest_date.iloc[1:]

Date_Valeur=datetime.strptime("2024-04-16", "%Y-%m-%d") # Date de Valorisation de notre portefeuil
date=Date_Valeur     
              
# j'ai ajouter cette ligne pour valoriser mon port au date au valeur
data_Valorisation_Date['Valorisation_à_la_date_'+ str(Date_Valeur)]= data_Valorisation_Date.apply(lambda row: Valorisation_BDT(row['txno']/100, row['nominal'], find_nearest_date(Date_Valeur, data) , Date_Valeur, row['dt/ech']), axis=1)                                            
for sem in range(104):
    date=date - timedelta(days=7)                                        
    data_Valorisation_Date['Valorisation_à_la_date_'+ str(date)]= data_Valorisation_Date.apply(lambda row: Valorisation_BDT(row['txno']/100, row['nominal'], find_nearest_date(date, data) , date, row['dt/ech']), axis=1)                                            

# %%
# exporter les données en format excel
data_Valorisation_Date.to_excel('Valorsiation du portefeuil.xlsx')

# %% [markdown]
# #### Calculon la VaR

# %% [markdown]
# ##### Var Historique

# %%
# sommer par date de valeur pour obtenr la valeur du portefeuil pour chaque date
BDT_sum_Val_Port=data_Valorisation_Date.filter(regex='^Valorisation_à_la_date_').sum()
BDT_sum_Val_Port.to_numpy()                                                            

# %%
BDT_sum_Val_Port_df = pd.DataFrame({'Date_De_Valeur': BDT_sum_Val_Port.index, 'Valeur': BDT_sum_Val_Port.values})
BDT_sum_Val_Port_df['P&L']= BDT_sum_Val_Port_df['Valeur'] - BDT_sum_Val_Port_df['Valeur'].shift(-1)
BDT_sum_Val_Port_df['Rendement']=np.log( BDT_sum_Val_Port_df['Valeur']/ BDT_sum_Val_Port_df['Valeur'].shift(1))
# Afficher le DataFrame
# BDT_sum_Val_Port_df

# %%
BDT_sum_Val_Port_df['P&L'].sort_values()

# %%
BDT_sum_Val_Port_df.to_excel("Valorisaton du portefeuil par date.xlsx")

# %%
VaR = BDT_sum_Val_Port_df['Rendement'].quantile(0.01)  # Le quantile médian (50e percentile)
VaR*17700776293.29

# %% [markdown]
# ##### VaR MonteCarlo

# %%
data_Simule=data_BDT.copy()
a,b,sigma = 0.04413108, 0.0849325 , 0.01102

# %%
# Function to simulate interest rates using CIR model
dt=1/365.25

def simulate_cir(rt_prev, a, b, sigma, dt,horizone = 1): 
    rt = rt_prev #0jrs
    for _ in range(int(horizone)) : 
        dWt = np.random.normal(0, np.sqrt(dt))   
        rt = rt + a * (b - rt) * dt + sigma * dWt 
    return rt # aprs h jours

# a, b, sigma= 0.028959627154054465, 0.13503538016652458, 0.01497951439088685
Date_Valeur = datetime.strptime("2024-04-16", "%Y-%m-%d")
date = Date_Valeur     
for scenario in range(1_000):
    rt=simulate_cir(0.03, a, b, sigma, dt,horizone=1)  # Initialize short rate for each scenario
    zc = [R_vasi(x, rt, a, b, sigma) for x in maturities]
    data_Simule['Valorisation_scenario_' + str(scenario)] = data_Simule.apply(lambda row: Valorisation_BDT(row['txno']/100, row['nominal'], zc, date, row['dt/ech']), axis=1)

# %%
zc

# %%
# export to excel
data_Simule.to_excel('Valorisation du portefeuil par simulation des taux (Vasicek_CIR).xlsx')

# %%
# sommer par date de valeur pour obtenr la valeur du portefeuil pour chaque date
BDT_sum_Val_Port_Simule=data_Simule.filter(regex='^Valorisation_scenario_').sum()
BDT_sum_Val_Port_Simule_df = pd.DataFrame({'Date_De_Valeur': BDT_sum_Val_Port_Simule.index, 'Valeur': BDT_sum_Val_Port_Simule.values})
BDT_sum_Val_Port_Simule_df['P&L']= BDT_sum_Val_Port_Simule_df['Valeur'] - 17700776293.29 
# Afficher le DataFrame
VaR = BDT_sum_Val_Port_Simule_df['P&L'].quantile(0.01)  # Le quantile 
VaR

# %%
BDT_sum_Val_Port_Simule_df.to_excel('BDT_sum_Val_Port_Simule_df.xlsx')

# %% [markdown]
# # ***************************************************************

# %% [markdown]
# Dans cette section On va utiliser la methode analytique pour calculer la VaR

# %% [markdown]
# ### Test de la normalité

# %%
dff=BDT_sum_Val_Port_df.copy()
plt.figure(figsize=(10, 6))
sns.histplot(dff['Rendement'], bins=50, kde=True, color='blue')
plt.title('Distribution des rendements')
plt.xlabel('Rendements')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# 3. Tests de normalité
shapiro_test = stats.shapiro(dff['Rendement'])
print("Test de Shapiro-Wilk (H0: la distribution est normale) : p-value =", shapiro_test.pvalue)

# 4. Ajustement de distributions
# Vous pouvez essayer différentes distributions et ajuster leurs paramètres pour trouver la meilleure adéquation.


# %% [markdown]
# there is not enough evidence to reject normality. so we consider that dist is normal

# %%
dff=dff.dropna()

# 1. Test de distribution de Student (t-distribution)
student_params = stats.t.fit(dff['Rendement'])
student_dist = stats.t(*student_params)
student_test = stats.kstest(dff['Rendement'], student_dist.cdf)
print("Test de Kolmogorov-Smirnov (Student) : p-value =", student_test.pvalue)

# 2. Test de distribution du khi-deux (chi-squared)
chi2_params = stats.chi2.fit(dff['Rendement'])
chi2_dist = stats.chi2(*chi2_params)
chi2_test = stats.kstest(dff['Rendement'], chi2_dist.cdf)
print("Test de Kolmogorov-Smirnov (Chi-squared) : p-value =", chi2_test.pvalue)


# %% [markdown]
# On peut aajouter que notre distribution est normal

# %%
from scipy.stats import jarque_bera

# Test de Jarque-Bera
jb_test_stat, jb_p_value = jarque_bera(dff['Rendement'].dropna())

print("Statistique de test de Jarque-Bera :", jb_test_stat)
print("p-value :", jb_p_value)

# Interprétation du test
if jb_p_value < 0.05:
    print("Les données ne suivent pas une distribution normale (rejet de l'hypothèse nulle)")
else:
    print("Les données suivent une distribution normale (non-rejet de l'hypothèse nulle)")


# %% [markdown]
# On a effectué deux tests : le test de Jarque-Bera et le test de Shapiro. Alors que le test de Shapiro suggère que notre distribution est normale, le test de Jarque-Bera indique le contraire.
# 
# Supposons que notre distribution suive une loi normale. Dès lors, déterminons les paramètres de cette loi.

# %%
# Calcul de la moyenne et de l'écart-type empiriques des rendements
mean_return = dff['Rendement'].mean()
std_return = dff['Rendement'].std()

print("Moyenne des rendements :", mean_return)
print("Écart-type des rendements :", std_return)


# %% [markdown]
# Pour déterminer la Value at Risk (VaR) au seuil de confiance de 99%, nous allon utiliser l'approche parametrique
# 
# Rappelons que la VaR représente la perte maximale attendue pour un niveau de confiance donné sur une certaine période. Pour une distribution normale, la VaR peut être calculée en utilisant la formule suivante :
# 
# $ \text{VaR}_{99\%} = \text{mean} + \text{std} \times \text{z-score}_{99\%} $
# 
# Où :
# - mean est la moyenne des rendements,
# - std est l'écart-type des rendements,
# - z-score_{99%} est le z-score correspondant au niveau de confiance de 99%.
# 
# Le z-score associé à un niveau de confiance de 99% est généralement de 2.33 pour une distribution normale.

# %%
# Calcul de la VaR au seuil de 99%
z_score_99 = 2.33  # Z-score pour un niveau de confiance de 99%
VaR_99 = mean_return + std_return * z_score_99
print("La VaR au seuil de 99% est :", VaR_99)
print("La VaR au seuil de 99% à l'horizon 1 jours est :", VaR_99/np.sqrt(5))

# %% [markdown]
# Il y a une autre chose a montrer est que si la dist n'est pas normale on va utiliser le quantile corrigé $w_{\alpha}$ defini par la formule suivante:  
# $$
# \boxed{w_{\alpha}=z_{\alpha}+\frac{1}{6}(z_{\alpha}^{2}-1)S+\frac{1}{24}(z_{\alpha}^{3}-3z_{\alpha})K-\frac{1}{36}(2z_{\alpha}^{3}-5z_{\alpha})S^{2}}
# $$
# 
# Le calcul de la Value-at-Risk se fait de manière similaire au calcul de la VaR dans le cas d’une loi 
# normale. Il suffit alors de remplacer $z-score_{99\%}$ par $w_α$.

# %%
dff=dff.dropna()
from scipy.stats import skew, kurtosis # type: ignore

# Calcul du coefficient de skewness
skewness = skew(dff['Rendement'])

# Calcul du coefficient de kurtosis
kurt = kurtosis(dff['Rendement'])

print("Coefficient de skewness (pour une distribution de la loi normale soit egal =0):", skewness)
print("Coefficient de kurtosis (pour une distribution de la loi normale soit egal =3):", kurt)  

# %%
def calculate_w(alpha, z_alpha, S, K):
    w_alpha = z_alpha + (1/6) * (z_alpha**2 - 1) * S + (1/24) * (z_alpha**3 - 3*z_alpha) * K - (1/36) * (2*z_alpha**3 - 5*z_alpha) * S**2
    return w_alpha

# Exemple d'utilisation avec des valeurs arbitraires pour alpha, z_alpha, S et K 
alpha = 0.99                                                 
z_alpha = 2.33  # Z-score pour un niveau de confiance de 99%
S = skewness  # Coefficient de skewness de vos données
K = kurt  # Coefficient de kurtosis de vos données

w_alpha = calculate_w(alpha, z_alpha, S, K)               
print("w_alpha :", w_alpha)

# %%
# Calcul de la VaR dans le cas où la distribution non normale                                
VaR_99 = mean_return + std_return * w_alpha                                
print("La VaR au seuil de 99% est :", VaR_99)                                
print("La VaR au seuil de 99% à l'horizon 1 jours est :",VaR_99/np.sqrt(5))                                

# %% [markdown]
# La VaR au seuil de 99% est : 0.012370793159665617
# La VaR au seuil de 99% à l'horizon 1 jours est : 0.004675720317299167

# %% [markdown]
# Dans ce cas on va utiliser la distribution de student

# %%
ddf['Rendement'] = ddf['Rendement'].fillna(0)

# Estimation de la moyenne et de l'écart type
mean =gmean(ddf['Rendement'])
std_dev = np.std(ddf['Rendement'])

# Degré de liberté
nu = len(ddf['Rendement']) - 1

# Calcul du quantile pour estimer les paramètres de la loi de Student
quantile = t.ppf(0.99, nu)

# Calcul des paramètres
student_mean = mean
student_std_dev = std_dev * np.sqrt((nu-2) / nu) * quantile

print('student_mean =', student_mean)
print('student_std_dev=', student_std_dev)

# %%
VaR_99 = student_mean + student_std_dev * quantile
VaR_99/np.sqrt(5)

# %%
ddf=BDT_sum_Val_Port_df.copy()
dates=[ddf.iloc[:,0].index[i] for i in range(105)]  # les indices correspond aux dates de valorisation de portefeuil

# %%
ddf['Valeur'][0]

# %%
17679101586*0.005345102073666928

# %% [markdown]
# -- VaR_Monte Carlo(vaR1) == 17210565.9239455
# 
# -- VaR_Historique(VaR2) == 94496602.54789686
# 
# -- VaR_Anlytique(VaR3) == 0.004675720317299167\*17679101586= 82662534.47725613 (Normal) 0.010686572402988058\*17679101586 188928999.11857(Student)

# %% [markdown]
# 
# $$
# \text{{VaR\_Anlytique(VaR3)}} = \left\{
# \begin{array}{ll}
# 82662534.47725613 & \text{{(Normal)}} \\
# 188928999.11857 & \text{{(Student)}}
# \end{array}
# \right.
# $$
# 

# %%
ddf['Valeur']

# %%
import matplotlib.pyplot as plt

# Supposons que vous ayez une liste de dates correspondant aux valeurs du portefeuille
# Assurez-vous de remplacer [...] par vos propres dates

# Supposons que vous ayez une liste de valeurs de portefeuille par date
portefeuille_valeurs = ddf['P&L']  # Assurez-vous de remplacer [...] par vos propres valeurs de portefeuille
dates=ddf['Date_De_Valeur']
# Supposons que vous ayez une valeur fixe pour la VaR
# VaR = 82662534.47725613  # Normal VaR
# VaR= -188928999.11857  # Student
# VaR = 94496602.54789686 # Historique
VaR = 1527742059.542604 #Monte Carlo
# Plot des valeurs de portefeuille
plt.plot(dates, portefeuille_valeurs, label='Portefeuille')

# Plot de la VaR (une droite horizontale)
plt.axhline(y=VaR, color='r', linestyle='--', label='VaR')

# Ajout de légendes et de titres
plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Evolution du Portefeuille et VaR')
plt.legend()

# Affichage du graphique
plt.show()

# %% [markdown]
# ## using maximum likelihood

# %% [markdown]
# Le modèle de CIR est le suivant: 
# $$
# dr_t = α(µ − rt)dt + \sqrt {r_t} \sigma dW_t
# 
# $$
# 
# \begin{equation}
# p(r_{t+\Delta t} | r_t; \theta, \Delta t) = c e^{-u-v} \left( \frac{v}{u} \right)^{ \frac{q}{2}} I_q(2\sqrt{uv}),
# \end{equation}
# 
# where
# \begin{align*}
# c &= \frac{2\alpha}{\sigma^2 (1 - e^{-\alpha \Delta t})}, \\
# u &= cr_te^{-\alpha \Delta t}, \\
# v &= cr_{t+\Delta t}, \\
# q &= \frac{2\alpha\mu}{\sigma^2} - 1,
# \end{align*}
# 
# and $I_q(2\sqrt{uv})$ is the modified Bessel function of the first kind and of order $q$.
# 

# %% [markdown]
# # *******************************

# %%
rt=df_c['1 J']
rt

# %% [markdown]
# #*************************************

# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import iv  # Modified Bessel function of the first kind

# # Supposons que rt soit la série temporelle des taux d'intérêt
rt = df_c['1 J'].values


# %%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import iv  # Modified Bessel function of the first kind

# # Supposons que rt soit la série temporelle des taux d'intérêt
rt = df_c['1 J'].values

N = len(rt)        
delta_t = 1 # Intervalle de temps (par exemple, 1 jour)        

# Fonction pour calculer les estimations initiales d'alpha et mu        
def initial_estimates(rt, delta_t):
    y = (rt[1:] - rt[:-1]) / np.sqrt(rt[:-1])
    X = np.vstack([1/np.sqrt(rt[:-1]), np.sqrt(rt[:-1]) * delta_t]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha = -beta[1] / delta_t
    mu = beta[0] / (beta[1] * delta_t)
    return alpha, mu

# Estimation initiale de sigma
def estimate_sigma(rt, alpha, mu, delta_t):
    residuals = (rt[1:] - rt[:-1] - alpha * (mu - rt[:-1]) * delta_t) / np.sqrt(rt[:-1])
    sigma = np.std(residuals)
    return sigma

# Fonction de log-vraisemblance
def log_likelihood(theta, rt, delta_t):
    alpha, mu, sigma = theta
    N = len(rt)
    c = 2 * alpha / (sigma**2 * (1 - np.exp(-alpha * delta_t)))
    q = 2 * alpha * mu / sigma**2 - 1
    
    log_lik = 0
    for i in range(N - 1):
        u = c * rt[i] * np.exp(-alpha * delta_t)
        v = c * rt[i+1]
        log_lik += (np.log(c) - u - v + 0.5 * q * np.log(v/u) + np.log(iv(q, 2 * np.sqrt(u * v))))
    return -log_lik

# Calculer les estimations initiales
# alpha_init, mu_init = 0.1, 0.05
# sigma_init = 0.02
alpha_init, mu_init=initial_estimates(rt, delta_t)

alpha_init, mu_init = 0.1137805121585627, 0.04311965977532673
sigma_init = 0.019999641509027735

# Estimations initiales
theta_init = [alpha_init, mu_init, sigma_init]

# Maximiser la log-vraisemblance
result = minimize(log_likelihood, theta_init, args=(rt, delta_t), bounds=((0, None), (0, None), (0, None)), method="L-BFGS-B")

# Estimations finales
alpha_est, mu_est, sigma_est = result.x

print(f"Estimation de alpha: {alpha_est}")
print(f"Estimation de mu: {mu_est}")
print(f"Estimation de sigma: {sigma_est}")

# %%
import numpy as np
import pandas as pd

# Supposons que rt soit la série temporelle des taux d'intérêt
rt = df_c['1 J'].values

N = len(rt)
delta_t = 1  # Intervalle de temps (par exemple, 1 jour)

# Calculer les sommes nécessaires
sum_rt = np.sum(rt[:-1])
sum_rt_next = np.sum(rt[1:])
sum_1_rt = np.sum(1 / rt[:-1])
sum_rt_rt_next = np.sum(rt[1:] / rt[:-1])
sum_rt_1_rt = np.sum(rt[:-1] / rt[:-1])

# Calculer alpha
numerator_alpha = N**2 - 2*N + 1 + sum_rt_next * sum_1_rt - sum_rt * sum_1_rt - (N - 1) * sum_rt_rt_next
denominator_alpha = (N**2 - 2*N + 1 - sum_rt * sum_1_rt) * delta_t

alpha = numerator_alpha / denominator_alpha

# Calculer mu

numerator_mu = (N - 1) * sum_rt_next - sum_rt_rt_next * sum_rt 
denominator_mu = N**2 - 2*N + 1 + sum_rt_next * sum_1_rt - sum_rt * sum_1_rt - (N - 1) * sum_rt_rt_next

mu = numerator_mu / denominator_mu

print(f"Estimation de alpha: {alpha}")
print(f"Estimation de mu: {mu}")


# %%
residuals = (rt[1:] - rt[:-1] - alpha * (mu - rt[:-1]) * delta_t) / np.sqrt(rt[:-1])

# Estimer sigma
sigma = np.std(residuals)
print(f"Estimation de sigma: {sigma}")

# %%

# Charger les données
data = pd.read_excel('data.xlsx')

# Convertir les dates en datetime
data['Date de référence'] = pd.to_datetime(data['Date de référence'])

# Trier les données par date
data = data.sort_values(by='Date de référence')

# Calculer les rendements log
data['1J_log_return'] = np.log(data['1 J'] / data['1 J'].shift(1))

# Retirer les NaN résultants du décalage
data = data.dropna()

# Afficher les premières lignes des données avec les rendements
data.head()


# %%
from scipy.optimize import minimize

# Fonction pour calculer la vraisemblance négative du modèle CIR
def neg_log_likelihood(params, returns):
    a, b, sigma = params
    n = len(returns)
    likelihood = -np.sum(-0.5 * (returns**2) / (sigma**2 * returns) - 0.5 * np.log(2 * np.pi * sigma**2 * returns))
    return -likelihood

# Initialiser les paramètres
initial_params = [0.1, 0.05, 0.02]

# Effectuer l'optimisation pour trouver les meilleurs paramètres
result = minimize(neg_log_likelihood, initial_params, args=(data['1J_log_return'],), method='L-BFGS-B', bounds=[(0, None), (0, None), (0, None)])

# Extraire les paramètres optimisés
a_opt, b_opt, sigma_opt = result.x

print(f"Paramètres optimisés du modèle CIR:\na = {a_opt}\nb = {b_opt}\nsigma = {sigma_opt}")


