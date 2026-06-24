
from __future__ import annotations

import math
from functools import lru_cache
from collections.abc import Callable

import numpy as np

###################################################
###################################################
############## Parâmetros de entrada ##############

Rm = 5            # Raio da Projeção Beam (m)
h = 3              # Altura da Antena (m)
h_user = 1.5        # Altura do Usuário (m)
freq = 10e9         # Frequência (Hz)
c = 3e8             # Velocidade da luz (m/s)
lambda_0 = c / freq # Comprimento de onda (m)
d0 = 1              # Distancia de referencia (m)
gamma_t = 1e9       # SNR transmitida padrao em escala linear
K = 1               # Fator de Rice
mu = 1              # Número de Clusters de Componentes Multipercurso
md = 2              # Coeficiente de Shadowing (1 = max shadowing e inf = sem shadowing)
delta = 2.5           # Exponente de Path Loss
N_amostras = 800_000 # Numero de amostras Monte Carlo
N_bins_histograma = 50 # Numero de bins do histograma da SNR
gamma_th_outage_db = 0 # Limiar de outage da SNR instantanea (dB)
gamma_th_outage = 10 ** (gamma_th_outage_db / 10) # Limiar de outage em escala linear
gamma_t_db_min = 60 # Menor SNR transmitida no grafico de outage (dB)
gamma_t_db_max = 110 # Maior SNR transmitida no grafico de outage (dB)
N_pontos_outage = 18 # Numero de pontos da curva de outage
K_curvas = [1, 5, 10, 15] # Valores de K para os graficos
K_curvas_outage = K_curvas # Valores de K para o grafico de outage
delta_curvas = [2, 2.5, 3, 3.5] # Valores de delta para os graficos
Rm_curvas = [2, 5, 10, 20] # Valores de Rm para os graficos
parametro_variado = "Rm" # Opcoes: "kappa", "delta" ou "Rm"
N_termos_serie = 40 # Numero de termos da serie shadowed kappa-mu
a_sep = 1 # Parametro a da Eq. 21 para BPSK coerente
b_sep = 2 # Parametro b da Eq. 21 para BPSK coerente

###################################################
###################################################
###################################################

class DistribuicaoPorPDF:
    """Adaptador esperado pelo NumericalInversePolynomial do SciPy."""

    def __init__(self, pdf: Callable[[float | np.ndarray], float | np.ndarray]):
        self._pdf = pdf

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._pdf(x)


def gerar_amostras_por_fdp(
    fdp: Callable[[float | np.ndarray], float | np.ndarray],
    n: int,
    dominio: tuple[float, float],
    *,
    semente: int | None = None,
    ordem: int = 5,
    u_resolution: float = 1e-10,
) -> np.ndarray:
    """Gera N amostras aleatorias distribuidas conforme uma FDP informada.

    A FDP deve ser nao negativa e integravel no intervalo `dominio`. Ela nao
    precisa estar normalizada: o metodo PINV do SciPy normaliza numericamente
    usando a integral calculada no dominio.

    Args:
        fdp: Funcao densidade de probabilidade f(x).
        n: Numero de amostras a gerar.
        dominio: Intervalo fechado (xmin, xmax) onde a FDP esta definida.
        semente: Semente opcional para reprodutibilidade.
        ordem: Ordem do polinomio usado pela aproximacao PINV.
        u_resolution: Tolerancia aproximada no eixo de probabilidades.

    Returns:
        Vetor NumPy com `n` amostras.
    """

    xmin, xmax = dominio
    try:
        from scipy.stats.sampling import NumericalInversePolynomial
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    distribuicao = DistribuicaoPorPDF(fdp)
    gerador = NumericalInversePolynomial(
        distribuicao,
        domain=(xmin, xmax),
        order=ordem,
        u_resolution=u_resolution,
        random_state=semente,
    )

    return gerador.rvs(size=n)


def gerar_posicao_usuario_no_beam(
    raio_maximo: float = Rm,
    altura_antena: float = h,
    altura_usuario: float = h_user,
) -> dict[str, float]:
    """Gera a posicao aleatoria do usuario no beam atendido.

    A posicao horizontal e uniforme na area circular do beam:
        theta ~ Uniforme(0, 2*pi)
        r = raio_maximo * sqrt(U), com U ~ Uniforme(0, 1)

    A distancia ate a pinching antenna e:
        D(r) = sqrt((altura_antena - altura_usuario)^2 + r^2)
    """

    r = raio_maximo * np.sqrt(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2 * np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    distancia = np.sqrt((altura_antena - altura_usuario) ** 2 + r**2)

    return {
        "r": r,
        "theta": theta,
        "x": x,
        "y": y,
        "distancia": distancia,
    }


def calcular_perda_percurso(
    distancia: float | np.ndarray,
    *,
    comprimento_onda: float = lambda_0,
    distancia_referencia: float = d0,
    expoente_perda: float = delta,
) -> float | np.ndarray:
    """Calcula rho_l(R), a perda de percurso dependente da distancia.

    rho_l(R) = (lambda_0 / (4*pi*d0))^2 * (D(R) / d0)^(-delta)
    """

    distancia = np.asarray(distancia)
    ganho_referencia = (comprimento_onda / (4 * np.pi * distancia_referencia)) ** 2
    perda_distancia = (distancia / distancia_referencia) ** (-expoente_perda)

    return ganho_referencia * perda_distancia


def calcular_snr_media_condicionada(
    distancia: float | np.ndarray,
    *,
    snr_transmitida: float = gamma_t,
    expoente_perda: float = delta,
) -> float | np.ndarray:

    perda_percurso = calcular_perda_percurso(
        distancia,
        expoente_perda=expoente_perda,
    )
    return snr_transmitida * perda_percurso


def calcular_coeficiente_ad(
    n: int,
    *,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
) -> float:
    """Calcula o coeficiente A_d(n; md, kappa, mu) da PDF do artigo."""

    try:
        from scipy.special import gamma, kv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    argumento_bessel = 2 * np.sqrt((md_param - 1) * kappa * mu_param)
    ordem_bessel = md_param - n

    numerador = (
        2
        * (md_param - 1) ** ((md_param + n) / 2)
        * kappa ** ((md_param + n) / 2)
        * (1 + kappa) ** (n + mu_param)
        * mu_param ** ((3 * n + md_param) / 2 + mu_param)
        * kv(ordem_bessel, argumento_bessel)
    )
    denominador = math.factorial(n) * gamma(md_param) * gamma(n + mu_param)

    return numerador / denominador


def fdp_envelope_shadowed_kappa_mu(
    x: float | np.ndarray,
    *,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    numero_termos: int = N_termos_serie,
) -> float | np.ndarray:
    """PDF do envelope |H| para o fading shadowed kappa-mu do artigo.

    A soma infinita da PDF e aproximada pelos primeiros `numero_termos`.
    Assume-se normalizacao E[|H|^2] = 1, como no artigo.
    """

    entrada_escalar = np.isscalar(x)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    pdf = np.zeros_like(x, dtype=float)
    x_positivo = x > 0

    if not np.any(x_positivo):
        return pdf[0] if entrada_escalar else pdf

    xp = x[x_positivo]
    expoente = np.exp(-(1 + kappa) * mu_param * xp**2)

    for n in range(numero_termos):
        ad = calcular_coeficiente_ad(
            n,
            md_param=md_param,
            kappa=kappa,
            mu_param=mu_param,
        )
        pdf[x_positivo] += 2 * ad * xp ** (2 * n + 2 * mu_param - 1) * expoente

    pdf = np.maximum(pdf, 0)
    return pdf[0] if entrada_escalar else pdf


@lru_cache(maxsize=64)
def calcular_massa_fdp_envelope(
    md_param: float,
    kappa: float,
    mu_param: float,
    numero_termos: int,
    limite_superior: float = 12,
) -> float:
    """Calcula a massa da PDF truncada usada pelo PINV.

    O PINV normaliza numericamente a PDF fornecida. Para comparar simulado e
    analitico com a mesma serie truncada, a Eq. 16 tambem deve ser normalizada
    pela massa dessa PDF truncada.
    """

    x = np.linspace(1e-8, limite_superior, 50000)
    fx = fdp_envelope_shadowed_kappa_mu(
        x,
        md_param=md_param,
        kappa=kappa,
        mu_param=mu_param,
        numero_termos=numero_termos,
    )
    massa = np.trapezoid(fx, x)
    return float(massa)


def gerar_amostras_fading_shadowed_kappa_mu(
    n: int,
    *,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    numero_termos: int = N_termos_serie,
    dominio_envelope: tuple[float, float] = (1e-8, 8),
    semente: int | None = None,
) -> np.ndarray:
    """Gera amostras do ganho de canal |H|^2 usando PINV.

    Primeiro sao geradas amostras do envelope |H| pela PDF do artigo. Depois,
    retorna-se o ganho de potencia |H|^2.
    """

    def fdp_envelope(x):
        return fdp_envelope_shadowed_kappa_mu(
            x,
            md_param=md_param,
            kappa=kappa,
            mu_param=mu_param,
            numero_termos=numero_termos,
        )

    envelope = gerar_amostras_por_fdp(
        fdp_envelope,
        n=n,
        dominio=dominio_envelope,
        semente=semente,
    )

    return envelope**2


def calcular_gamma_barra_0(
    *,
    snr_transmitida: float = gamma_t,
    comprimento_onda: float = lambda_0,
    distancia_referencia: float = d0,
    expoente_perda: float = delta,
) -> float:
    """Calcula gamma_barra_0 da equacao 9 com alpha = 0."""

    return (
        snr_transmitida
        * (comprimento_onda / (4 * np.pi)) ** 2
        * distancia_referencia ** (expoente_perda - 2)
    )


def fdp_snr_instantanea_equacao_16(
    gamma: float | np.ndarray,
    *,
    raio_maximo: float = Rm,
    altura_antena: float = h,
    altura_usuario: float = h_user,
    snr_transmitida: float = gamma_t,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    expoente_perda: float = delta,
    numero_termos: int = N_termos_serie,
) -> float | np.ndarray:
    """PDF incondicional da SNR instantanea conforme a equacao 16.

    Esta expressao considera usuario uniformemente distribuido na area circular
    de raio Rm e h_w = 1, com alpha = 0 na perda do guia de onda.
    """

    try:
        from scipy.special import gammainc, gamma as gamma_func
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    entrada_escalar = np.isscalar(gamma)
    gamma_snr = np.atleast_1d(np.asarray(gamma, dtype=float))
    pdf = np.zeros_like(gamma_snr, dtype=float)
    gamma_positivo = gamma_snr > 0

    if not np.any(gamma_positivo):
        return pdf[0] if entrada_escalar else pdf

    gp = gamma_snr[gamma_positivo]
    gamma_barra_0 = calcular_gamma_barra_0(
        snr_transmitida=snr_transmitida,
        expoente_perda=expoente_perda,
    )

    l = altura_antena - altura_usuario
    limite_inferior_distancia = abs(l) ** expoente_perda
    limite_superior_distancia = (l**2 + raio_maximo**2) ** (expoente_perda / 2)

    fator_fading = (1 + kappa) * mu_param
    c_l = fator_fading * limite_inferior_distancia / gamma_barra_0
    c_u = fator_fading * limite_superior_distancia / gamma_barra_0

    soma = np.zeros_like(gp, dtype=float)
    for n in range(numero_termos):
        s = n + mu_param + 2 / expoente_perda
        ad = calcular_coeficiente_ad(
            n,
            md_param=md_param,
            kappa=kappa,
            mu_param=mu_param,
        )
        gama_inf_u = gamma_func(s) * gammainc(s, c_u * gp)
        gama_inf_l = gamma_func(s) * gammainc(s, c_l * gp)
        soma += ad * (gama_inf_u - gama_inf_l) / (fator_fading**s)

    massa_envelope = calcular_massa_fdp_envelope(
        md_param,
        kappa,
        mu_param,
        numero_termos,
    )

    pdf[gamma_positivo] = (
        2
        / (expoente_perda * raio_maximo**2)
        * gamma_barra_0 ** (2 / expoente_perda)
        * gp ** (-(1 + 2 / expoente_perda))
        * soma
        / massa_envelope
    )

    pdf = np.maximum(pdf, 0)
    return pdf[0] if entrada_escalar else pdf


def funcao_q_equacao_17(
    n: int,
    a: float,
    b: float,
    c: float | np.ndarray,
) -> float | np.ndarray:
    """Calcula Q(n; a, b, c) da equacao 17 do artigo."""

    try:
        from scipy.special import gammainc, gamma as gamma_func
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    entrada_escalar = np.isscalar(c)
    c = np.atleast_1d(np.asarray(c, dtype=float))
    resultado = np.zeros_like(c, dtype=float)
    c_positivo = c > 0

    if not np.any(c_positivo):
        return resultado[0] if entrada_escalar else resultado

    cp = c[c_positivo]
    s1 = n + a
    s2 = n + a + 2 / b

    gamma_inc_1 = gamma_func(s1) * gammainc(s1, cp)
    gamma_inc_2 = gamma_func(s2) * gammainc(s2, cp)

    resultado[c_positivo] = gamma_inc_1 - gamma_inc_2 / cp ** (2 / b)
    return resultado[0] if entrada_escalar else resultado


def cdf_snr_instantanea_equacao_18(
    gamma: float | np.ndarray,
    *,
    raio_maximo: float = Rm,
    altura_antena: float = h,
    altura_usuario: float = h_user,
    snr_transmitida: float = gamma_t,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    expoente_perda: float = delta,
    numero_termos: int = N_termos_serie,
) -> float | np.ndarray:
    """CDF incondicional da SNR instantanea conforme a equacao 18."""

    entrada_escalar = np.isscalar(gamma)
    gamma_snr = np.atleast_1d(np.asarray(gamma, dtype=float))
    cdf = np.zeros_like(gamma_snr, dtype=float)
    gamma_positivo = gamma_snr > 0

    if not np.any(gamma_positivo):
        return cdf[0] if entrada_escalar else cdf

    gp = gamma_snr[gamma_positivo]
    gamma_barra_0 = calcular_gamma_barra_0(
        snr_transmitida=snr_transmitida,
        expoente_perda=expoente_perda,
    )

    l = altura_antena - altura_usuario
    c_l = (1 + kappa) * mu_param * abs(l) ** expoente_perda / gamma_barra_0
    c_u = (
        (1 + kappa)
        * mu_param
        * (l**2 + raio_maximo**2) ** (expoente_perda / 2)
        / gamma_barra_0
    )

    fator_fading = (1 + kappa) * mu_param
    massa_envelope = calcular_massa_fdp_envelope(
        md_param,
        kappa,
        mu_param,
        numero_termos,
    )

    soma = np.zeros_like(gp, dtype=float)
    for n in range(numero_termos):
        ad = calcular_coeficiente_ad(
            n,
            md_param=md_param,
            kappa=kappa,
            mu_param=mu_param,
        )
        denominador = fator_fading ** (n + mu_param + 2 / expoente_perda)
        q_u = funcao_q_equacao_17(n, mu_param, expoente_perda, c_u * gp)
        q_l = funcao_q_equacao_17(n, mu_param, expoente_perda, c_l * gp)
        soma += ad / denominador * (
            c_u ** (2 / expoente_perda) * q_u
            - c_l ** (2 / expoente_perda) * q_l
        )

    cdf[gamma_positivo] = (
        gamma_barra_0 ** (2 / expoente_perda)
        / raio_maximo**2
        * soma
        / massa_envelope
    )

    cdf = np.clip(cdf, 0, 1)
    return cdf[0] if entrada_escalar else cdf


def gerar_amostras_snr_instantanea(
    n: int,
    *,
    raio_maximo: float = Rm,
    altura_antena: float = h,
    altura_usuario: float = h_user,
    snr_transmitida: float = gamma_t,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    expoente_perda: float = delta,
    numero_termos: int = N_termos_serie,
    dominio_envelope: tuple[float, float] = (1e-8, 8),
) -> np.ndarray:
    """Gera amostras Monte Carlo da SNR instantanea Gamma.

    Gamma = gamma_barra(R) * |H|^2

    A posicao do usuario segue distribuicao uniforme na area circular.
    """

    r = raio_maximo * np.sqrt(np.random.uniform(0, 1, size=n))
    distancia = np.sqrt((altura_antena - altura_usuario) ** 2 + r**2)
    snr_media = calcular_snr_media_condicionada(
        distancia,
        snr_transmitida=snr_transmitida,
        expoente_perda=expoente_perda,
    )
    ganho_fading = gerar_amostras_fading_shadowed_kappa_mu(
        n,
        md_param=md_param,
        kappa=kappa,
        mu_param=mu_param,
        numero_termos=numero_termos,
        dominio_envelope=dominio_envelope,
    )

    return snr_media * ganho_fading


def db_para_linear(valor_db: float | np.ndarray) -> float | np.ndarray:
    """Converte valor em dB para escala linear."""

    return 10 ** (np.asarray(valor_db) / 10)


def gerar_amostras_ganho_equivalente(
    n: int,
    *,
    raio_maximo: float = Rm,
    altura_antena: float = h,
    altura_usuario: float = h_user,
    md_param: float = md,
    kappa: float = K,
    mu_param: float = mu,
    expoente_perda: float = delta,
    numero_termos: int = N_termos_serie,
    dominio_envelope: tuple[float, float] = (1e-8, 8),
) -> np.ndarray:
    """Gera amostras de G = rho_l(R)|H|^2.

    A SNR instantanea para uma SNR transmitida gamma_t e:
        Gamma = gamma_t * G
    """

    r = raio_maximo * np.sqrt(np.random.uniform(0, 1, size=n))
    distancia = np.sqrt((altura_antena - altura_usuario) ** 2 + r**2)
    perda_percurso = calcular_perda_percurso(
        distancia,
        expoente_perda=expoente_perda,
    )
    ganho_fading = gerar_amostras_fading_shadowed_kappa_mu(
        n,
        md_param=md_param,
        kappa=kappa,
        mu_param=mu_param,
        numero_termos=numero_termos,
        dominio_envelope=dominio_envelope,
    )

    return perda_percurso * ganho_fading


def calcular_probabilidade_outage(
    ganho_equivalente: np.ndarray,
    snr_transmitida: float,
    *,
    limiar_snr: float = gamma_th_outage,
) -> float:
    """Calcula P_out = Pr(Gamma < gamma_th) por Monte Carlo."""

    snr_instantanea = snr_transmitida * ganho_equivalente
    return np.mean(snr_instantanea < limiar_snr)


def simular_curva_outage(
    ganho_equivalente: np.ndarray,
    snr_transmitida_db: np.ndarray,
    *,
    limiar_snr: float = gamma_th_outage,
) -> np.ndarray:
    """Simula a curva de outage para varios valores de gamma_t em dB."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_probabilidade_outage(
                ganho_equivalente,
                snr_t,
                limiar_snr=limiar_snr,
            )
            for snr_t in snr_transmitida_linear
        ]
    )


def calcular_probabilidade_outage_analitica(
    limiar_snr: float,
    snr_transmitida: float,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> float:
    """Calcula P_out = F_Gamma(gamma_th) pela CDF da Eq. 18."""

    probabilidade = cdf_snr_instantanea_equacao_18(
        limiar_snr,
        raio_maximo=raio_maximo,
        snr_transmitida=snr_transmitida,
        kappa=kappa,
        expoente_perda=expoente_perda,
    )
    return float(np.clip(probabilidade, 0, 1))


def calcular_curva_outage_analitica(
    snr_transmitida_db: np.ndarray,
    *,
    limiar_snr: float = gamma_th_outage,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> np.ndarray:
    """Calcula a curva analitica de outage, sem analise assintotica."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_probabilidade_outage_analitica(
                limiar_snr,
                snr_t,
                raio_maximo=raio_maximo,
                kappa=kappa,
                expoente_perda=expoente_perda,
            )
            for snr_t in snr_transmitida_linear
        ]
    )


def calcular_capacidade_ergodica(
    ganho_equivalente: np.ndarray,
    snr_transmitida: float,
) -> float:
    """Calcula C = E[log2(1 + Gamma)] por Monte Carlo."""

    snr_instantanea = snr_transmitida * ganho_equivalente
    return float(np.mean(np.log2(1 + snr_instantanea)))


def simular_curva_capacidade_ergodica(
    ganho_equivalente: np.ndarray,
    snr_transmitida_db: np.ndarray,
) -> np.ndarray:
    """Simula a curva de capacidade ergodica para varios gamma_t em dB."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_capacidade_ergodica(ganho_equivalente, snr_t)
            for snr_t in snr_transmitida_linear
        ]
    )


def calcular_capacidade_ergodica_analitica(
    snr_transmitida: float,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> float:
    """Calcula a capacidade ergodica pela Eq. 23 do artigo."""

    try:
        from scipy.integrate import quad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    def integrando(gamma_valor):
        return np.log2(1 + gamma_valor) * fdp_snr_instantanea_equacao_16(
            gamma_valor,
            raio_maximo=raio_maximo,
            snr_transmitida=snr_transmitida,
            kappa=kappa,
            expoente_perda=expoente_perda,
        )

    capacidade, _ = quad(
        integrando,
        0,
        np.inf,
        epsabs=1e-7,
        epsrel=1e-5,
        limit=150,
    )
    return float(max(capacidade, 0))


def calcular_curva_capacidade_ergodica_analitica(
    snr_transmitida_db: np.ndarray,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> np.ndarray:
    """Calcula a curva analitica de capacidade ergodica."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_capacidade_ergodica_analitica(
                snr_t,
                raio_maximo=raio_maximo,
                kappa=kappa,
                expoente_perda=expoente_perda,
            )
            for snr_t in snr_transmitida_linear
        ]
    )


def calcular_sep_bpsk(
    ganho_equivalente: np.ndarray,
    snr_transmitida: float,
) -> float:
    """Calcula SEP media para BPSK coerente por Monte Carlo."""

    try:
        from scipy.special import erfc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    snr_instantanea = snr_transmitida * ganho_equivalente
    sep_instantanea = 0.5 * erfc(np.sqrt(snr_instantanea))
    return float(np.mean(sep_instantanea))


def simular_curva_sep_bpsk(
    ganho_equivalente: np.ndarray,
    snr_transmitida_db: np.ndarray,
) -> np.ndarray:
    """Simula a curva SEP media para BPSK coerente."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_sep_bpsk(ganho_equivalente, snr_t)
            for snr_t in snr_transmitida_linear
        ]
    )


def calcular_sep_bpsk_analitica(
    snr_transmitida: float,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
    parametro_a: float = a_sep,
    parametro_b: float = b_sep,
) -> float:
    """Calcula a SEP media pela Eq. 21 do artigo.

    Para BPSK coerente, usamos a = 1 e b = 2.
    """

    try:
        from scipy.integrate import quad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    def integrando(gamma_valor):
        return (
            gamma_valor ** (-0.5)
            * cdf_snr_instantanea_equacao_18(
                gamma_valor,
                raio_maximo=raio_maximo,
                snr_transmitida=snr_transmitida,
                kappa=kappa,
                expoente_perda=expoente_perda,
            )
            * np.exp(-parametro_b * gamma_valor / 2)
        )

    integral, _ = quad(
        integrando,
        0,
        np.inf,
        epsabs=1e-9,
        epsrel=1e-5,
        limit=150,
    )
    sep = parametro_a * np.sqrt(parametro_b) / (2 * np.sqrt(2 * np.pi)) * integral
    return float(np.clip(sep, 0, 0.5))


def calcular_sep_bpsk_analitica_via_pdf(
    snr_transmitida: float,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> float:
    """Calcula SEP BPSK via PDF, mantida apenas para validacao numerica."""

    try:
        from scipy.integrate import quad
        from scipy.special import erfc
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Esta funcao requer SciPy. Instale com: pip install scipy"
        ) from exc

    def integrando(gamma_valor):
        pe = 0.5 * erfc(np.sqrt(gamma_valor))
        return pe * fdp_snr_instantanea_equacao_16(
            gamma_valor,
            raio_maximo=raio_maximo,
            snr_transmitida=snr_transmitida,
            kappa=kappa,
            expoente_perda=expoente_perda,
        )

    sep, _ = quad(
        integrando,
        0,
        np.inf,
        epsabs=1e-9,
        epsrel=1e-5,
        limit=150,
    )
    return float(np.clip(sep, 0, 0.5))


def calcular_curva_sep_bpsk_analitica(
    snr_transmitida_db: np.ndarray,
    *,
    raio_maximo: float = Rm,
    kappa: float = K,
    expoente_perda: float = delta,
) -> np.ndarray:
    """Calcula a curva analitica de SEP media para BPSK coerente."""

    snr_transmitida_linear = db_para_linear(snr_transmitida_db)
    return np.array(
        [
            calcular_sep_bpsk_analitica(
                snr_t,
                raio_maximo=raio_maximo,
                kappa=kappa,
                expoente_perda=expoente_perda,
            )
            for snr_t in snr_transmitida_linear
        ]
    )


def obter_configuracoes_curvas(
    parametro: str = parametro_variado,
) -> tuple[str, list[dict[str, float]], str]:
    """Monta as configuracoes das curvas para o parametro escolhido."""

    if parametro == "kappa":
        configuracoes = [{"kappa": valor, "delta": delta, "Rm": Rm} for valor in K_curvas]
        return "K", configuracoes, "kappa"
    if parametro == "delta":
        configuracoes = [{"kappa": K, "delta": valor, "Rm": Rm} for valor in delta_curvas]
        return "$\\delta$", configuracoes, "delta"
    if parametro == "Rm":
        configuracoes = [{"kappa": K, "delta": delta, "Rm": valor} for valor in Rm_curvas]
        return "$R_m$", configuracoes, "Rm"

    configuracoes = [{"kappa": valor, "delta": delta, "Rm": Rm} for valor in K_curvas]
    return "K", configuracoes, "kappa"


def formatar_valor_parametro(nome_parametro: str, valor: float) -> str:
    """Formata o valor variado para legenda."""

    if nome_parametro == "Rm":
        return f"{valor:g} m"
    return f"{valor:g}"


def montar_texto_parametros(configuracao: dict[str, float], nome_parametro: str) -> str:
    """Monta a caixa de parametros fixos mostrada no grafico."""

    linhas = [
        f"$m_d$ = {md}",
        f"$\\mu$ = {mu}",
        f"$\\gamma_{{th}}$ = {gamma_th_outage_db} dB",
    ]

    if nome_parametro != "kappa":
        linhas.insert(2, f"$K$ = {configuracao['kappa']:g}")
    if nome_parametro != "delta":
        linhas.insert(-1, f"$\\delta$ = {configuracao['delta']:g}")
    if nome_parametro != "Rm":
        linhas.insert(-1, f"$R_m$ = {configuracao['Rm']:g} m")

    return "\n".join(linhas)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    posicao_usuario = gerar_posicao_usuario_no_beam()
    print("posicao aleatoria do usuario no beam:")
    print(f"r = {posicao_usuario['r']:.4f} m")
    print(f"theta = {posicao_usuario['theta']:.4f} rad")
    print(f"x = {posicao_usuario['x']:.4f} m")
    print(f"y = {posicao_usuario['y']:.4f} m")
    print(f"D(r) = {posicao_usuario['distancia']:.4f} m")
    print(f"lambda_0 = {lambda_0:.6f} m")

    perda_percurso = calcular_perda_percurso(posicao_usuario["distancia"])
    print(f"rho_l(R) = {perda_percurso:.6e}")

    snr_media = calcular_snr_media_condicionada(posicao_usuario["distancia"])
    print(f"gamma_t = {gamma_t:.6e}")
    print(f"gamma_barra(R) = {snr_media:.6e}")

    amostras_snr = gerar_amostras_snr_instantanea(n=N_amostras)
    amostras_snr = amostras_snr[amostras_snr > 0]

    limite_inferior = max(np.quantile(amostras_snr, 0.001), 1e-12)
    limite_superior = np.quantile(amostras_snr, 0.999)
    gamma_eixo = np.logspace(
        np.log10(limite_inferior),
        np.log10(limite_superior),
        1000,
    )
    fdp_snr = fdp_snr_instantanea_equacao_16(gamma_eixo)
    bins = np.logspace(
        np.log10(limite_inferior),
        np.log10(limite_superior),
        N_bins_histograma,
    )

    plt.figure(figsize=(9, 5))
    plt.hist(
        amostras_snr,
        bins=bins,
        density=True,
        alpha=0.45,
        label="Histograma Monte Carlo da SNR",
    )
    plt.plot(
        gamma_eixo,
        fdp_snr,
        "r",
        linewidth=2,
        label="FDP da SNR instantanea - Eq. 16",
    )
    plt.xscale("log")
    plt.title("FDP da SNR instantanea")
    plt.xlabel("SNR instantanea, gamma")
    plt.ylabel("Densidade de probabilidade")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fdp_snr_histograma_equacao_16.png", dpi=300)
    plt.close()

    gamma_t_db = np.linspace(gamma_t_db_min, gamma_t_db_max, N_pontos_outage)
    nome_legenda, configuracoes_curvas, nome_parametro = obter_configuracoes_curvas()
    texto_parametros = montar_texto_parametros(configuracoes_curvas[0], nome_parametro)

    plt.figure(figsize=(9, 5))
    for configuracao in configuracoes_curvas:
        valor_curva = configuracao[nome_parametro]
        rotulo_valor = formatar_valor_parametro(nome_parametro, valor_curva)
        ganho_equivalente = gerar_amostras_ganho_equivalente(
            N_amostras,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )
        prob_outage = simular_curva_outage(
            ganho_equivalente,
            gamma_t_db,
            limiar_snr=gamma_th_outage,
        )
        prob_outage_analitica = calcular_curva_outage_analitica(
            gamma_t_db,
            limiar_snr=gamma_th_outage,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )

        linha, = plt.semilogy(
            gamma_t_db,
            prob_outage_analitica,
            "-",
            linewidth=2,
            label=f"Analitico, {nome_legenda} = {rotulo_valor}",
        )
        plt.semilogy(
            gamma_t_db,
            prob_outage,
            "x",
            color=linha.get_color(),
            markersize=6,
            mew=1.5,
            linestyle="None",
            label=f"Simulado, {nome_legenda} = {rotulo_valor}",
        )
    plt.gca().text(
        0.03,
        0.05,
        texto_parametros,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.title("Probabilidade de Outage")
    plt.xlabel("SNR transmitida, gamma_t (dB)")
    plt.ylabel("Probabilidade de outage")
    plt.ylim(1 / N_amostras, 1)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("probabilidade_outage.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    for configuracao in configuracoes_curvas:
        valor_curva = configuracao[nome_parametro]
        rotulo_valor = formatar_valor_parametro(nome_parametro, valor_curva)
        ganho_equivalente = gerar_amostras_ganho_equivalente(
            N_amostras,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )
        capacidade_simulada = simular_curva_capacidade_ergodica(
            ganho_equivalente,
            gamma_t_db,
        )
        capacidade_analitica = calcular_curva_capacidade_ergodica_analitica(
            gamma_t_db,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )

        linha, = plt.plot(
            gamma_t_db,
            capacidade_analitica,
            "-",
            linewidth=2,
            label=f"Analitico, {nome_legenda} = {rotulo_valor}",
        )
        plt.plot(
            gamma_t_db,
            capacidade_simulada,
            "x",
            color=linha.get_color(),
            markersize=6,
            mew=1.5,
            linestyle="None",
            label=f"Simulado, {nome_legenda} = {rotulo_valor}",
        )

    plt.gca().text(
        0.03,
        0.95,
        texto_parametros,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.title("Capacidade Ergodica")
    plt.xlabel("SNR transmitida, gamma_t (dB)")
    plt.ylabel("Capacidade ergodica (bits/s/Hz)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("capacidade_ergodica.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    for configuracao in configuracoes_curvas:
        valor_curva = configuracao[nome_parametro]
        rotulo_valor = formatar_valor_parametro(nome_parametro, valor_curva)
        ganho_equivalente = gerar_amostras_ganho_equivalente(
            N_amostras,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )
        sep_simulada = simular_curva_sep_bpsk(
            ganho_equivalente,
            gamma_t_db,
        )
        sep_analitica = calcular_curva_sep_bpsk_analitica(
            gamma_t_db,
            raio_maximo=configuracao["Rm"],
            kappa=configuracao["kappa"],
            expoente_perda=configuracao["delta"],
        )

        linha, = plt.semilogy(
            gamma_t_db,
            sep_analitica,
            "-",
            linewidth=2,
            label=f"Analitico, {nome_legenda} = {rotulo_valor}",
        )
        plt.semilogy(
            gamma_t_db,
            sep_simulada,
            "x",
            color=linha.get_color(),
            markersize=6,
            mew=1.5,
            linestyle="None",
            label=f"Simulado, {nome_legenda} = {rotulo_valor}",
        )

    plt.gca().text(
        0.03,
        0.05,
        texto_parametros,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.title("SEP Media - BPSK")
    plt.xlabel("SNR transmitida, gamma_t (dB)")
    plt.ylabel("SEP media")
    plt.ylim(1 / N_amostras, 1)
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("sep_bpsk.png", dpi=300)
    plt.close()

    print("Grafico salvo: fdp_snr_histograma_equacao_16.png")
    print("Grafico salvo: probabilidade_outage.png")
    print("Grafico salvo: capacidade_ergodica.png")
    print("Grafico salvo: sep_bpsk.png")
