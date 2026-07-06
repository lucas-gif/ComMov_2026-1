"""
Simulacao de um canal MIMO cell-free.

Preambulo com os parametros basicos
"""

import numpy as np
import matplotlib.pyplot as plt

# Frequencia da portadora
fc = 3.0e9  # Hz

# Largura de banda
Bw = 20.0e6  # Hz

# Figura de ruido
Fn_dB = 9.0  # dB
Fn = 10 ** (Fn_dB / 10)  # valor linear

# Alturas
hAP = 15.0  # metros
hUE = 1.65

# Temperatura local
T = 296.15  # Kelvin

# Tamanho da area
Lx = 1000.0  # metros
Ly = 1000.0  

# Numero de APs e UEs
M = 100
K = 10

# Parametros dos loops Monte Carlo
Nbc = 100
Ncf = 300

# Parametros de coerencia
tau_cf = 20
tau_c = 200

# Potencias fisicas usadas para obter as SNRs normalizadas do artigo
P_p = 100e-3  # W
P_d = 200e-3  # W

# Desvio padrao do sombreamento
sigma_sf = 8.0  # dB

# Constante de Boltzmann
k0 = 1.381e-23  # J/K

# Velocidade da luz
c = 3.0e8  # m/s


def calcula_potencia_ruido(k0, T, Bw, Fn):
    """Calcula a potencia do ruido: sigma_w^2 = k0*T*Bw*Fn."""
    sigma_w_2 = k0 * T * Bw * Fn
    return sigma_w_2


def gera_posicoes_aps(M, Lx, Ly, hAP):
    """Gera as posicoes dos APs no formato [xAP, yAP, hAP]."""
    xAP = np.random.uniform(-Lx / 2, Lx / 2, M)
    yAP = np.random.uniform(-Ly / 2, Ly / 2, M)
    zAP = hAP * np.ones(M)
    pAP = np.column_stack((xAP, yAP, zAP))
    return pAP


def gera_posicoes_ues(K, Lx, Ly, hUE):
    """Gera as posicoes dos UEs no formato [xUE, yUE, hUE]."""
    xUE = np.random.uniform(-Lx / 2, Lx / 2, K)
    yUE = np.random.uniform(-Ly / 2, Ly / 2, K)
    zUE = hUE * np.ones(K)
    pUE = np.column_stack((xUE, yUE, zUE))
    return pUE


def calcula_matriz_distancias(pAP, pUE):
    """Calcula a matriz d_mk de distancias entre APs e UEs."""
    d = np.linalg.norm(pAP[:, np.newaxis, :] - pUE[np.newaxis, :, :], axis=2)
    return d


def calcula_perda_espaco_livre_1m(fc, c):
    """Calcula PL_FS(1 m, fc) em dB."""
    PL_FS_1m = 20 * np.log10((4 * np.pi * fc) / c)
    return PL_FS_1m


def calcula_desvanescimento_larga_escala(d, fc, c, sigma_sf):
    """Calcula a perda em dB e o ganho linear Omega para o modelo CI."""
    PL_FS_1m = calcula_perda_espaco_livre_1m(fc, c)
    X_sf = np.random.normal(0, sigma_sf, d.shape)
    Omega_dB = PL_FS_1m + 28 * np.log10(d) + X_sf
    Omega = 10 ** (-Omega_dB / 10)
    return Omega_dB, Omega


def gera_desvanescimento_pequena_escala(M, K):
    """Gera h_mk = hI_mk + j*hQ_mk com hI,hQ ~ N(0, 1/2)."""
    hI = np.random.normal(0, np.sqrt(1 / 2), (M, K))
    hQ = np.random.normal(0, np.sqrt(1 / 2), (M, K))
    h = hI + 1j * hQ
    return h


def calcula_coeficientes_canal(Omega, h):
    """Calcula G = sqrt(Omega)*h."""
    G = np.sqrt(Omega) * h
    return G


def gera_sequencias_piloto(K, tau_cf):

    if tau_cf < K:
        raise ValueError("Para pilotos ortogonais, e necessario tau_cf >= K.")

    phi = np.eye(tau_cf, K, dtype=complex)
    return phi


def simula_recepcao_piloto(G, phi, rho_p):

    M, K = G.shape
    tau_cf = phi.shape[0]
    
    # Ruido ao longo dos tau_cf simbolos
    # Para cada símbolo e cada dimensão (real e imaginária): sigma_w_2 / 2
    w_real = np.random.normal(0, np.sqrt(1 / 2), (tau_cf, M))
    w_imag = np.random.normal(0, np.sqrt(1 / 2), (tau_cf, M))
    w_p = w_real + 1j * w_imag  # (tau_cf, M)
    
    # Sinal piloto recebido em cada AP (sem ruido)
    # y_p,m[t] = sqrt(tau_cf * rho_p) * sum_k(g[m,k] * phi[t,k])
    # Forma matricial: y_p = sqrt(tau_cf * rho_p) * phi @ G.T, (tau_cf, M)
    y_p_sem_ruido = np.sqrt(tau_cf * rho_p) * phi @ G.T  # (tau_cf, M)
    
    # Adicionar ruído
    y_p_total = y_p_sem_ruido + w_p  # (tau_cf, M)
    
    # Projetar com sequencias piloto: y_p:m,k = phi_k^H * y_p,m
    # Forma matricial: y_p = phi^H @ y_p_total, resultado (K, M)
    y_p = (phi.conj().T @ y_p_total).T  # (M, K)
    
    return y_p


def estima_canal_mmse(y_p, Omega, tau_cf, rho_p):
  
    M, K = y_p.shape
    
    # Calcular os coeficientes MMSE para cada elemento (m,k)
    numerador = np.sqrt(tau_cf * rho_p) * Omega
    denominador = tau_cf * rho_p * Omega + 1
    c = numerador / denominador
    
    # Estimar o canal
    g_hat = c * y_p
    
    return g_hat, c


def calcula_coeficientes_mmse(Omega, tau_cf, rho_p):

    numerador = np.sqrt(tau_cf * rho_p) * Omega
    denominador = tau_cf * rho_p * Omega + 1
    c = numerador / denominador
    return c


def calcula_gamma(Omega, c_mmse, tau_cf, rho_p):
    gamma = np.sqrt(tau_cf * rho_p) * Omega * c_mmse
    return gamma


def calcula_coeficientes_controle_potencia(gamma):
    soma_gamma_por_ap = np.sum(gamma, axis=1, keepdims=True)
    eta = 1 / soma_gamma_por_ap
    eta = np.repeat(eta, gamma.shape[1], axis=1)
    return eta


def calcula_sinr_downlink(Omega, gamma, eta, rho_d):
    numerador = rho_d * (np.sum(np.sqrt(eta) * gamma, axis=0) ** 2)

    K = Omega.shape[1]
    denominador = np.zeros(K)
    for k in range(K):
        denominador[k] = rho_d * np.sum(eta * gamma * Omega[:, [k]]) + 1

    SINR = numerador / denominador
    return SINR


def calcula_taxa_alcancavel(SINR, Bw, tau_cf, tau_c):
    prelog = (1 - tau_cf / tau_c) / 2
    taxa = Bw * prelog * np.log2(1 + SINR) / 1e6
    return taxa


def calcula_ecdf(valores):
    x = np.sort(np.ravel(valores))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def simula_downlink(M, K, Nbc, Ncf):
    SINR_amostras = []
    taxa_amostras = []

    tau_cf_local = max(tau_cf, K)
    phi_local = gera_sequencias_piloto(K, tau_cf_local)

    for _ in range(Ncf):
        pAP_local = gera_posicoes_aps(M, Lx, Ly, hAP)
        pUE_local = gera_posicoes_ues(K, Lx, Ly, hUE)
        d_local = calcula_matriz_distancias(pAP_local, pUE_local)
        _, Omega_local = calcula_desvanescimento_larga_escala(d_local, fc, c, sigma_sf)

        for _ in range(Nbc):
            h_local = gera_desvanescimento_pequena_escala(M, K)
            G_local = calcula_coeficientes_canal(Omega_local, h_local)
            y_p_local = simula_recepcao_piloto(G_local, phi_local, rho_p)
            _, c_mmse_local = estima_canal_mmse(
                y_p_local, Omega_local, tau_cf_local, rho_p
            )

            gamma_local = calcula_gamma(Omega_local, c_mmse_local, tau_cf_local, rho_p)
            eta_local = calcula_coeficientes_controle_potencia(gamma_local)
            SINR_local = calcula_sinr_downlink(
                Omega_local, gamma_local, eta_local, rho_d
            )
            taxa_local = calcula_taxa_alcancavel(SINR_local, Bw, tau_cf_local, tau_c)

            SINR_amostras.extend(SINR_local)
            taxa_amostras.extend(taxa_local)

    return np.array(SINR_amostras), np.array(taxa_amostras)


def plota_cdf_por_M(valores_M, K_fixo, Nbc, Ncf):
    """Plota CDFs de SINR e taxa para diferentes valores de M."""
    resultados = {}
    for M_atual in valores_M:
        resultados[M_atual] = simula_downlink(M_atual, K_fixo, Nbc, Ncf)

    plt.figure()
    for M_atual in valores_M:
        SINR_atual, taxa_atual = resultados[M_atual]
        x_sinr, y_sinr = calcula_ecdf(10 * np.log10(SINR_atual))
        plt.plot(x_sinr, y_sinr, label=f"M = {M_atual}")
    plt.xlabel("SINR (dB)")
    plt.ylabel("ECDF")
    plt.grid(True)
    plt.legend()
    plt.title("CDF empirica da SINR para diferentes valores de M")

    plt.figure()
    for M_atual in valores_M:
        SINR_atual, taxa_atual = resultados[M_atual]
        x_taxa, y_taxa = calcula_ecdf(taxa_atual)
        plt.plot(x_taxa, y_taxa, label=f"M = {M_atual}")
    plt.xlabel("Taxa alcancavel (Mbits/s)")
    plt.ylabel("ECDF")
    plt.grid(True)
    plt.legend()
    plt.title("CDF empirica da taxa alcancavel para diferentes valores de M")


def plota_cdf_por_K(M_fixo, valores_K, Nbc, Ncf):
    """Plota CDFs de SINR e taxa para diferentes valores de K."""
    resultados = {}
    for K_atual in valores_K:
        resultados[K_atual] = simula_downlink(M_fixo, K_atual, Nbc, Ncf)

    plt.figure()
    for K_atual in valores_K:
        SINR_atual, taxa_atual = resultados[K_atual]
        x_sinr, y_sinr = calcula_ecdf(10 * np.log10(SINR_atual))
        plt.plot(x_sinr, y_sinr, label=f"K = {K_atual}")
    plt.xlabel("SINR (dB)")
    plt.ylabel("ECDF")
    plt.grid(True)
    plt.legend()
    plt.title("CDF empirica da SINR para diferentes valores de K")

    plt.figure()
    for K_atual in valores_K:
        SINR_atual, taxa_atual = resultados[K_atual]
        x_taxa, y_taxa = calcula_ecdf(taxa_atual)
        plt.plot(x_taxa, y_taxa, label=f"K = {K_atual}")
    plt.xlabel("Taxa alcancavel (Mbits/s)")
    plt.ylabel("ECDF")
    plt.grid(True)
    plt.legend()
    plt.title("CDF empirica da taxa alcancavel para diferentes valores de K")


# Potencia do ruido
sigma_w_2 = calcula_potencia_ruido(k0, T, Bw, Fn)

# SNRs normalizadas do artigo
rho_p = P_p / sigma_w_2
rho_d = P_d / sigma_w_2

# Posicoes dos APs e UEs
pAP = gera_posicoes_aps(M, Lx, Ly, hAP)
pUE = gera_posicoes_ues(K, Lx, Ly, hUE)

# Matriz de distancias entre APs e UEs
d = calcula_matriz_distancias(pAP, pUE)

# Desvanescimento em larga escala
Omega_dB, Omega = calcula_desvanescimento_larga_escala(d, fc, c, sigma_sf)

# Normalizar Omega para uma escala razoável
# (Omega original tem valores muito grandes, normalizamos por um fator de referência)
# Desvanescimento em pequena escala
h = gera_desvanescimento_pequena_escala(M, K)

# Matriz de coeficientes de canal verdadeiros (usando Omega normalizado)
G = calcula_coeficientes_canal(Omega, h)

# ===== FASE DE ESTIMACAO DO CANAL =====

# Gerar sequencias piloto ortogonais
phi = gera_sequencias_piloto(K, tau_cf)

# Simular a recepcao do sinal piloto em cada AP
y_p = simula_recepcao_piloto(G, phi, rho_p)

# Estimar o canal usando MMSE
g_hat, c_mmse = estima_canal_mmse(y_p, Omega, tau_cf, rho_p)

# Calcular o erro de estimacao
erro_estimacao = np.abs(g_hat - G)
erro_medio = np.mean(erro_estimacao)
mse = np.mean(np.abs(g_hat - G)**2)  # Mean Squared Error

# ===== FASE DE TRANSMISSAO DE DADOS (DOWNLINK) =====

gamma = calcula_gamma(Omega, c_mmse, tau_cf, rho_p)
eta = calcula_coeficientes_controle_potencia(gamma)
SINR = calcula_sinr_downlink(Omega, gamma, eta, rho_d)
taxa = calcula_taxa_alcancavel(SINR, Bw, tau_cf, tau_c)

# ===== GRAFICOS CDF =====

valores_M = [100, 150, 200]
valores_K = [10, 20, 30]

plota_cdf_por_M(valores_M, K, Nbc, Ncf)
plt.figure(1)
plt.savefig("cdf_sinr_por_M.png", dpi=300, bbox_inches="tight")
plt.figure(2)
plt.savefig("cdf_taxa_por_M.png", dpi=300, bbox_inches="tight")

plota_cdf_por_K(M, valores_K, Nbc, Ncf)
plt.figure(3)
plt.savefig("cdf_sinr_por_K.png", dpi=300, bbox_inches="tight")
plt.figure(4)
plt.savefig("cdf_taxa_por_K.png", dpi=300, bbox_inches="tight")

plt.close("all")
