import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import timedelta, datetime
import warnings
from PIL import Image

from streamlit_option_menu import option_menu
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calcul de la VaR", page_icon="🌍", layout="wide")

def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Calculations"],
            icons=["house", "eye"],
            menu_icon="cast",
            default_index=0
        )
    return selected

selected = sideBar()
# Charger et redimensionner l'image
img = Image.open("bp.png")
img = img.resize((200, 200))  # Ajustez la taille selon vos besoins

# Afficher l'image redimensionnée dans la barre latérale
st.sidebar.image(img, caption="")

# VasiceK parameters
a, b, sigma = 0.04413108, 0.0849325, 0.01102

# Import the zero-coupon data
url = 'https://raw.githubusercontent.com/medcharet/Projet-de-Fin-d-Etude/main/data_zc.xlsx'

# Lire le fichier Excel
data_zc = pd.read_excel(url)
df = data_zc.copy()
maturities = np.array([1/365.25, 7/365.25, 1/12, 2/12, 3/12, 6/12, 9/12, 1, 2, 3, 4, 5, 6, 
                       7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

# CIR model functions
def A(x, a, b, sigma):
    gamma = np.sqrt(a**2 + 2 * sigma**2)
    numerator = 2 * gamma * np.exp((a + gamma) * (x) / 2)
    denominator = (a + gamma) * (np.exp(gamma * (x)) - 1) + 2 * gamma
    return (numerator / denominator) ** (2 * a * b / sigma**2)

def B(x, a, sigma):
    gamma = np.sqrt(a**2 + 2 * sigma**2)
    numerator = 2 * (np.exp(gamma * (x)) - 1)
    denominator = (a + gamma) * (np.exp(gamma * (x)) - 1) + 2 * gamma
    return numerator / denominator

def P(x, rt, a, b, sigma):
    return A(x, a, b, sigma) * np.exp(-B(x, a, sigma) * rt)

def R(x, r, a, b, sigma):
    return np.log(P(x, r, a, b, sigma)) / (-x) if x != 0 else r

def P_vasi(x, rt, a, b, sigma):
    B_x = (1 - np.exp(-a * x)) / a
    A_x = np.exp(((B_x - x) * (a**2 * b - (sigma**2)/2)) / (a**2) - ((sigma**2 * B_x**2) / (4 * a)))
    return A_x * np.exp(-B_x * rt)

def R_vasi(x, r, a, b, sigma):
    return np.log(P_vasi(x, r, a, b, sigma)) / (-x) if x != 0 else r

def find_nearest_date(date, data_zc):
    nearest_date = data_zc.iloc[(data_zc['Date de référence'] - date).abs().idxmin()]
    return nearest_date.iloc[1:]

def Valorisation_BDT(tx, N, r, d_val, d_ech):
    date_ver_c = []
    date_ver_c.append(d_ech)
    x = d_ech
    while x > d_val:
        date_ver_c.append(x)
        x = x - timedelta(days=365.25)
    date_ver_c.sort()
    mat = [(date - d_val).days / 365.25 for date in date_ver_c]
    zc = interp1d(maturities, r, kind='linear', fill_value='extrapolate')
    valeur = 0
    for i in range(len(mat)):
        t = mat[i]
        valeur += (N * tx / (1 + zc(t)) ** t)
    valeur += N / (1 + zc(mat[-1])) ** mat[-1]
    return valeur

def Valorisation_BDT_Sim(tx, N, rt, d_val, d_ech):
    date_ver_c = []
    date_ver_c.append(d_ech)
    x = d_ech
    while x > d_val:
        date_ver_c.append(x)
        x = x - timedelta(days=365.25)
    date_ver_c.sort()
    mat = [(date - d_val).days / 365.25 for date in date_ver_c]
    valeur = 0
    def zc(t):
        return R_vasi(t, rt, a, b, sigma)
    for i in range(len(mat)):
        t = mat[i]
        valeur += N * tx / (1 + zc(t)) ** t
    valeur += N / (1 + zc(mat[-1])) ** mat[-1]
    return valeur

Date_Valeur = datetime.strptime("2024-04-16", "%Y-%m-%d")
dates = df['Date de référence'].apply(lambda x: x.to_pydatetime())

def var_historical(data, confidence_level):
    data_Valorisation_Date = data.copy()
    data_Valorisation_Date['dt/val'] = data_Valorisation_Date['dt/val'].apply(lambda x: x.to_pydatetime())
    data_Valorisation_Date['dt/ech'] = data_Valorisation_Date['dt/ech'].apply(lambda x: x.to_pydatetime())
    date = Date_Valeur
    while date >= dates[67]:
        data_Valorisation_Date['Valorisation_à_la_date_' + str(date)] = data_Valorisation_Date.apply(
            lambda row: Valorisation_BDT(row['txno'] / 100, row['nominal'], find_nearest_date(date, data_zc), date, row['dt/ech']), axis=1)
        date = date - timedelta(days=1)
    BDT_sum_Val_Port = data_Valorisation_Date.filter(regex='^Valorisation_à_la_date_').sum()
    BDT_sum_Val_Port_df = pd.DataFrame({'Date_De_Valeur': BDT_sum_Val_Port.index, 'Valeur': BDT_sum_Val_Port.values})
    BDT_sum_Val_Port_df['P&L'] = BDT_sum_Val_Port_df['Valeur'] - BDT_sum_Val_Port_df['Valeur'].shift(-1)
    BDT_sum_Val_Port_df['Rendement'] = np.log(BDT_sum_Val_Port_df['Valeur'] / BDT_sum_Val_Port_df['Valeur'].shift(-1))
    VaR_Hist = BDT_sum_Val_Port_df['P&L'].quantile(1 - confidence_level)
    return VaR_Hist, BDT_sum_Val_Port_df

def var_variance_covariance(data, confidence_level):
    mean_return = np.mean(data)
    std_return = np.std(data)
    var = abs(mean_return - std_return * np.percentile(data, (1 - confidence_level) * 100))
    return var

dt = 1/365
z_matrix = []
def simulate_cir_vas(rt_prev, a, b, sigma, dt, horizone=1):
    rt = rt_prev
    for _ in range(int(horizone)):
        dWt = np.random.normal(0, np.sqrt(dt))
        rt = rt + a * (b - rt) * dt + np.sqrt(rt) * sigma * dWt
    return rt

Date_Valeur = datetime.strptime("2024-04-16", "%Y-%m-%d")
date = Date_Valeur

def var_monte_carlo(data, confidence_level, num_simulations):
    data_Simule = data.copy()
    for scenario in range(num_simulations):
        rt = simulate_cir_vas(0.031, a, b, sigma, dt, horizone=1)
        zc = [R_vasi(x, rt, a, b, sigma) for x in maturities]
        z_matrix.append(zc)
        data_Simule['Valorisation_scenario_' + str(scenario)] = data_Simule.apply(lambda row: Valorisation_BDT_Sim(row['txno']/100, row['nominal'], rt, date, row['dt/ech']), axis=1)
    BDT_sum_Val_Port_Simule = data_Simule.filter(regex='^Valorisation_scenario_').sum()
    BDT_sum_Val_Port_Simule_df = pd.DataFrame({'Date_De_Valeur': BDT_sum_Val_Port_Simule.index, 'Valeur': BDT_sum_Val_Port_Simule.values})
    V0 = 18329481999.063927
    BDT_sum_Val_Port_Simule_df['P&L'] = BDT_sum_Val_Port_Simule_df['Valeur'] - V0
    BDT_sum_Val_Port_Simule_df['Rendement'] = np.log(BDT_sum_Val_Port_Simule_df['Valeur'] / V0)
    VaR_MonteCa = BDT_sum_Val_Port_Simule_df['Rendement'].quantile(1 - confidence_level) * V0
    return VaR_MonteCa, BDT_sum_Val_Port_Simule_df

# Streamlit interface
st.title('Value at Risk (VaR) Calculator')

if selected == "Home":
    st.header("Home")
    st.write("Bienvenue sur la page d'accueil!")
    st.write("Cette application permet de calculer la Value at Risk (VaR) en utilisant trois méthodes différentes :")
    st.write("1. Simulation Historique")
    st.write("2. Variance-Covariance")
    st.write("3. Simulation Monte Carlo")
    st.write("Veuillez sélectionner 'Calculations' dans le menu pour effectuer les calculs de VaR.")

elif selected == "Calculations":
    st.header("Calculations")
    st.write("Téléchargez votre jeu de données pour commencer les calculs de VaR.")

    uploaded_file = st.file_uploader('Upload your dataset', type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
        except:
            data = pd.read_excel(uploaded_file)

        required_columns = ['txno', 'nominal', 'dt/val', 'dt/ech']
        if not all(column in data.columns for column in required_columns):
            st.error(f'The uploaded dataset must contain the following columns: {", ".join(required_columns)}.')
        else:
            st.write('Dataset loaded successfully!')
            st.write(data.head())

            method = st.selectbox('Select VaR calculation method',
                                  ['Historical Simulation', 'Variance-Covariance'])

            confidence_level = st.slider('Select confidence level', 0.90, 0.99, 0.95)

            if st.button('Calculate VaR(Historique/Covarinance)'):
                if method == 'Historical Simulation':
                    var, BDT_sum_Val_Port_df = var_historical(data, confidence_level)
                    st.write(f'The calculated VaR at {confidence_level * 100}% confidence level using {method} is: {var}')
                    st.write('Historical Simulation Portfolio Values:')
                    st.write(BDT_sum_Val_Port_df)
                else:
                    returns = data['returns']  # Assuming the dataset has a column named 'returns'
                    var = var_variance_covariance(returns, confidence_level)
                    st.write(f'The calculated VaR at {confidence_level * 100}% confidence level using {method} is: {var}')
                    
            num_simulations = st.number_input('Choisir le Number of Simulations', min_value=100, max_value=10000, value=6000)
            if st.button('Calculate VaR par MonteCarlo'):
                var, BDT_sum_Val_Port_Simule_df = var_monte_carlo(data, confidence_level, num_simulations)
                st.write(f'The calculated VaR at {confidence_level * 100}% confidence level using {method} is: {var}')
                st.write('Monte Carlo Simulation Portfolio Values:')
                st.write(BDT_sum_Val_Port_Simule_df)
                st.write(f'The calculated VaR at {confidence_level * 100}% confidence level using MonteCarlo is: {var}')
            


# To run the app, save this script and execute: streamlit run your_script_name.py       
                                                                                        
