from dataclasses import dataclass
from math import acos, atan2, cos, degrees, exp, log, log10, pi, radians, sin, sqrt
from pathlib import Path
import random

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

#################################################################
#################################################################
#################################################################
# Parametros Gerais
fc = 3e9    # Frequencia de portadora (Hz)
scn = "InD"   # Cenario de simulacao ("UMi", "UMa" e "InD")
n = 100       # Numero de componentes multipercursos
T_pulse = 1e-7

# Estacao de Base (Base Station)
h_bs = 3
bs_pos = (0.0, 0.0, h_bs)

# Estacao Movel (User Terminal)
h_ut = 1
ut_pos = (7, 6, h_ut)

ut_speed = 1
ut_vector = (1.0, 1.0, 0.0)

#################################################################
#################################################################
#################################################################

@dataclass(frozen=True)
class LargeScaleParameter:
    mu: float | None
    sigma: float | None
    log_value: float | None
    linear_value: float | None
    unit: str


@dataclass(frozen=True)
class MultipathProfile:
    delays: list[float]
    powers: list[float]
    magnitudes: list[float]


@dataclass(frozen=True)
class AngularPowerProfile:
    arrival_angles: list[float]
    powers: list[float]


@dataclass(frozen=True)
class ScenarioConfig:
    los: bool
    fc: float
    h_bs: float
    h_ut: float
    d2d: float
    d3d: float
    scn: str
    spd: float
    vut: tuple[float, float, float]
    theta: float
    phi: float
    theta_l: float
    phi_l: float
    pr_los: float
    large_scale_params: dict[str, LargeScaleParameter]
    multipath_profile: MultipathProfile
    azimuth_angular_power_profile: AngularPowerProfile
    elevation_angular_power_profile: AngularPowerProfile
    multipath_unit_vectors: list[tuple[float, float, float]]
    doppler_shifts_hz: list[float]


def calculate_distances(
    ut_pos: tuple[float, float, float],
    bs_pos: tuple[float, float, float],
) -> tuple[float, float]:
    x_ut, y_ut, z_ut = ut_pos
    x_bs, y_bs, z_bs = bs_pos

    d2d = sqrt((x_ut - x_bs) ** 2 + (y_ut - y_bs) ** 2)
    d3d = sqrt((x_ut - x_bs) ** 2 + (y_ut - y_bs) ** 2 + (z_ut - z_bs) ** 2)

    return d2d, d3d


def calculate_angles(
    ut_pos: tuple[float, float, float],
    bs_pos: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    x_ut, y_ut, z_ut = ut_pos
    x_bs, y_bs, z_bs = bs_pos

    d2d, _ = calculate_distances(ut_pos, bs_pos)

    dx = x_ut - x_bs
    dy = y_ut - y_bs
    dz = z_ut - z_bs
    _, d3d = calculate_distances(ut_pos, bs_pos)

    # Angulos de saida: direcao da BS para o UT.
    # Convencao polar adotada:
    # theta = 0°   -> +z
    # theta = 90°  -> horizonte
    # theta = 180° -> -z
    phi = atan2(dy, dx)
    theta = acos(dz / d3d)

    # Angulos de chegada: direcao oposta, do UT para a BS.
    phi_l = atan2(-dy, -dx)
    theta_l = acos(-dz / d3d)

    return phi, phi_l, theta, theta_l


def calculate_los_probability(scn: str, d2d: float, h_ut: float) -> float:
    scenario = scn.strip().lower()

    # Probabilidade LOS associada a cada cenario.
    if scenario == "umi":
        if d2d <= 18:
            pr_los = 1.0
        else:
            pr_los = 18 / d2d + exp(-d2d / 36) * (1 - 18 / d2d)

    if scenario == "uma":
        if d2d <= 18:
            pr_los = 1.0
        else:
            if h_ut <= 13:
                c_h_ut = 0.0
            elif h_ut <= 23:
                c_h_ut = ((h_ut - 13) / 10) ** 1.5
            else:
                raise ValueError("Para UMa, h_ut deve ser menor ou igual a 23 m.")

            a = 18 / d2d + exp(-d2d / 63) * (1 - 18 / d2d)
            b = 1 + c_h_ut * (5 / 4) * (d2d / 100) ** 3 * exp(-d2d / 150)
            pr_los = a * b
            #print(f"Probabilidade de Visada Direta {pr_los*100} %")

    if scenario == "ind":
        if d2d <= 5:
            pr_los = 1.0
        elif d2d <= 49:
            pr_los = exp(-(d2d - 5) / 70.8)
        else:
            pr_los = exp(-(d2d - 49) / 211.7) * 0.54

    return pr_los


def calculate_large_scale_parameters(
    scn: str,
    fc: float,
    los: bool,
) -> dict[str, LargeScaleParameter]:
    scenario = scn.strip().lower()
    fc_ghz = fc / 1e9
    log_fc = log10(fc_ghz)
    log_1_plus_fc = log10(1 + fc_ghz)
    condition = "los" if los else "nlos"

    if scenario == "umi":
        table = {
            "los": {
                "DS": (-0.24 * log_1_plus_fc - 7.14, 0.38, "s"),
                "ASD": (-0.05 * log_1_plus_fc + 1.21, 0.41, "graus"),
                "ASA": (-0.08 * log_1_plus_fc + 1.73, 0.014 * log_1_plus_fc + 0.28, "graus"),
                "ZSA": (-0.10 * log_1_plus_fc + 0.73, -0.04 * log_1_plus_fc + 0.34, "graus"),
                "K": (9.0, 5.0, "linear"),
                "r_tau": (3.0, None, "adimensional"),
                "SF": (3.0, None, "dB"),
            },
            "nlos": {
                "DS": (-0.24 * log_1_plus_fc - 6.83, 0.16 * log_1_plus_fc + 0.28, "s"),
                "ASD": (-0.23 * log_1_plus_fc + 1.53, 0.11 * log_1_plus_fc + 0.33, "graus"),
                "ASA": (-0.08 * log_1_plus_fc + 1.81, 0.05 * log_1_plus_fc + 0.30, "graus"),
                "ZSA": (-0.04 * log_1_plus_fc + 0.92, -0.07 * log_1_plus_fc + 0.41, "graus"),
                "K": (None, None, "linear"),
                "r_tau": (2.1, None, "adimensional"),
                "SF": (3.0, None, "dB"),
            },
        }
    elif scenario == "uma":
        table = {
            "los": {
                "DS": (-6.955 - 0.0963 * log_fc, 0.66, "s"),
                "ASD": (1.06 + 0.1114 * log_fc, 0.28, "graus"),
                "ASA": (1.81, 0.20, "graus"),
                "ZSA": (0.95, 0.16, "graus"),
                "K": (9.0, 3.5, "linear"),
                "r_tau": (2.5, None, "adimensional"),
                "SF": (3.0, None, "dB"),
            },
            "nlos": {
                "DS": (-6.28 - 0.204 * log_fc, 0.39, "s"),
                "ASD": (1.50 - 0.1144 * log_fc, 0.28, "graus"),
                "ASA": (2.08 - 0.27 * log_fc, 0.11, "graus"),
                "ZSA": (-0.3236 * log_fc + 1.512, 0.16, "graus"),
                "K": (None, None, "linear"),
                "r_tau": (2.3, None, "adimensional"),
                "SF": (3.0, None, "dB"),
            },
        }
    elif scenario == "ind":
        table = {
            "los": {
                "DS": (-0.01 * log_1_plus_fc - 7.692, 0.18, "s"),
                "ASD": (1.60, 0.18, "graus"),
                "ASA": (-0.19 * log_1_plus_fc + 1.781, 0.12 * log_1_plus_fc + 0.119, "graus"),
                "ZSA": (-0.26 * log_1_plus_fc + 1.44, -0.04 * log_1_plus_fc + 0.264, "graus"),
                "K": (7.0, 4.0, "linear"),
                "r_tau": (3.6, None, "adimensional"),
                "SF": (6.0, None, "dB"),
            },
            "nlos": {
                "DS": (-0.28 * log_1_plus_fc - 7.173, 0.10 * log_1_plus_fc + 0.055, "s"),
                "ASD": (1.62, 0.25, "graus"),
                "ASA": (-0.11 * log_1_plus_fc + 1.863, 0.12 * log_1_plus_fc + 0.059, "graus"),
                "ZSA": (-0.15 * log_1_plus_fc + 1.387, -0.09 * log_1_plus_fc + 0.746, "graus"),
                "K": (None, None, "linear"),
                "r_tau": (3.0, None, "adimensional"),
                "SF": (3.0, None, "dB"),
            },
        }
    else:
        raise ValueError('Cenario deve ser "UMi", "UMa" ou "InD".')

    params = {}

    for name, (mu, sigma, unit) in table[condition].items():
        if mu is None:
            params[name] = LargeScaleParameter(
                mu=None,
                sigma=None,
                log_value=None,
                linear_value=None,
                unit=unit,
            )
            continue

        if sigma is None:
            params[name] = LargeScaleParameter(
                mu=mu,
                sigma=None,
                log_value=None,
                linear_value=mu,
                unit=unit,
            )
            continue

        log_value = random.gauss(mu, sigma)

        if name == "K":
            linear_value = 10 ** (log_value / 10)
        else:
            linear_value = 10 ** log_value
            if name in ("ASD", "ASA"):
                linear_value = min(linear_value, 104)
            elif name in ("ZSD", "ZSA"):
                linear_value = min(linear_value, 52)


        params[name] = LargeScaleParameter(
            mu=mu,
            sigma=sigma,
            log_value=log_value,
            linear_value=linear_value,
            unit=unit,
        )

    return params


def generate_multipath_delays(
    n_components: int,
    delay_spread: float,
    r_tau: float,
) -> list[float]:
    
    # Esse é o Lambda da Distribuição Exponencial
    mean_delay = r_tau * delay_spread
    
    # Gerando amostras exponencialmente distribuídas
    raw_delays = [random.expovariate(1 / mean_delay) for _ in range(n_components)]
    
    # Garantindo aqui, conforme orientado pelo professor, a amostra com tempo mínimo estará na origem
    min_delay = min(raw_delays)
    delays = [delay - min_delay for delay in raw_delays]
    
    # Deixando na ordem...
    delays.sort()

    return delays


def calculate_multipath_powers(
    delays: list[float],
    delay_spread: float,
    r_tau: float,
    k_factor: float | None,
    los: bool,
    shadow_std_db: float,
) -> list[float]:
    
    # Criando um vetor vazio
    preliminary_powers = []

    # Tendo os delay (atrasos) exponencialmente distribuidos, vamos calcular sua magnitude
    for delay in delays:
        # Per cluster shadowing std ζ [dB]
        
        shadowing_db = random.gauss(0, shadow_std_db)
        delay_decay = exp(-delay * (r_tau - 1) / (r_tau * delay_spread))
        shadowing_factor = 10 ** (-shadowing_db / 10)
        # Vou adicionando, a cada laço da repetição um elemento an²
        preliminary_powers.append(delay_decay * shadowing_factor)

    if los and k_factor is not None:
        
        # Estamos somando todas as componentes maiores que 1
        diffuse_power_sum = sum(preliminary_powers[1:])
        
        # Se tiver visada direta, aí definimos o primeiro elemento do vetor em powers
        # Estou definindo o valor da primeira componente propositalmente com este valor
        powers = [k_factor / (k_factor + 1)]
        
        # Supondo um fator de rice sempre maior que um, sei que o quociente 1/k+1 é sempre menor que k/k+1
        # Além do mais estou garantindo que todo o somatório resulte em 1 pois (Kr/Kr+1)+(1/Kr+1) = 1    
        
        for preliminary_power in preliminary_powers[1:]:
            powers.append(
                preliminary_power / diffuse_power_sum / (k_factor + 1)
            )
    
    # Caso não tenha visada direta...
    else:
        total_power = sum(preliminary_powers)
        powers = [power / total_power for power in preliminary_powers]

    return powers


def generate_multipath_profile(
    large_scale_params: dict[str, LargeScaleParameter],
    los: bool,
    n_components: int,
) -> MultipathProfile:
    delay_spread = large_scale_params["DS"].linear_value
    r_tau = large_scale_params["r_tau"].linear_value
    k_factor = large_scale_params["K"].linear_value
    shadow_std_db = large_scale_params["SF"].linear_value

    delays = generate_multipath_delays(n_components, delay_spread, r_tau)
    powers = calculate_multipath_powers(
        delays=delays,
        delay_spread=delay_spread,
        r_tau=r_tau,
        k_factor=k_factor,
        los=los,
        shadow_std_db=shadow_std_db,
    )
    magnitudes = [sqrt(power) for power in powers]

    return MultipathProfile(
        delays=delays,
        powers=powers,
        magnitudes=magnitudes,
    )


def plot_multipath_profile(
    profile: MultipathProfile,
    delay_spread: float,
    los: bool,
) -> None:
    delays_ns = [delay * 1e9 for delay in profile.delays]
    max_delay_ns = max(max(delays_ns), 1e-12)
    delay_spread_ns = delay_spread * 1e9
    los_text = "Sim" if los else "Nao"
    output_path = Path(__file__).with_name("multipath_profile.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    positive_magnitudes = [magnitude for magnitude in profile.magnitudes if magnitude > 0]
    min_magnitude = min(positive_magnitudes)
    max_magnitude = max(positive_magnitudes)
    y_min = max(min_magnitude / 10, max_magnitude / 1e7)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    colors = ["#2B1397"] * len(delays_ns)
    linewidths = [1.35] * len(delays_ns)

    if los:
        colors[0] = "#D62728"
        linewidths[0] = 2.2

    ax.vlines(
        delays_ns,
        ymin=y_min,
        ymax=profile.magnitudes,
        colors=colors,
        linewidth=linewidths,
    )
    ax.scatter(
        delays_ns,
        profile.magnitudes,
        marker="^",
        s=[42 if los and index == 0 else 24 for index in range(len(delays_ns))],
        color=colors,
        zorder=3,
    )

    if los:
        ax.plot(
            [],
            [],
            color="#D62728",
            marker="^",
            linewidth=2.2,
            label="Componente LOS",
        )

    ax.plot(
        [],
        [],
        color="#2B1397",
        marker="^",
        linewidth=1.35,
        label="Componentes dispersas",
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_min, top=max_magnitude * 1.5)
    ax.set_xlim(left=-0.02 * max_delay_ns, right=1.02 * max_delay_ns)
    ax.set_xlabel("Atraso (ns)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Componentes Multipercurso")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.98,
        0.82,
        f"DS = {delay_spread_ns:.3f} ns\nLOS = {los_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def wrap_angle_degrees(angle: float) -> float:
    return (angle + 180) % 360 - 180


def generate_azimuth_angular_power_profile(
    profile: MultipathProfile,
    asa: float,
    los_angle: float,
    los: bool,
) -> AngularPowerProfile:
    max_power = max(profile.powers)
    los_angle_deg = degrees(los_angle)
    azimuth_arrival_angles = []

    for power in profile.powers:

        # Calculando o ângulos iniciais phi_n duas linhas
        power_ratio = min(power / max_power, 1.0)
        initial_offset = 1.42 * asa * sqrt(-log(power_ratio))


        random_sign = random.choice([-1, 1])
        random_fluctuation = random.gauss(0, asa / 7)
        angle = random_sign * initial_offset + random_fluctuation + los_angle_deg
        azimuth_arrival_angles.append(wrap_angle_degrees(angle))

    if los:
        azimuth_arrival_angles[0] = wrap_angle_degrees(los_angle_deg)

    return AngularPowerProfile(arrival_angles=azimuth_arrival_angles, powers=profile.powers)


def plot_azimuth_angular_power_profile(
    profile: AngularPowerProfile,
    asa: float,
    los: bool,
) -> None:
    output_path = Path(__file__).with_name("angular_power_spectrum_azimuth.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    positive_powers = [power for power in profile.powers if power > 0]
    min_power = min(positive_powers)
    max_power = max(positive_powers)
    r_min = max(min_power / 10, max_power / 1e7)
    angles_rad = [radians(angle) for angle in profile.arrival_angles]

    colors = ["#2B1397"] * len(profile.arrival_angles)
    linewidths = [1.35] * len(profile.arrival_angles)

    if los:
        colors[0] = "#D62728"
        linewidths[0] = 2.2

    fig, ax = plt.subplots(figsize=(7, 7), dpi=140, subplot_kw={"projection": "polar"})

    for angle, power, color, linewidth in zip(
        angles_rad,
        profile.powers,
        colors,
        linewidths,
    ):
        ax.plot([angle, angle], [r_min, power], color=color, linewidth=linewidth)

    ax.scatter(
        angles_rad,
        profile.powers,
        s=[42 if los and index == 0 else 24 for index in range(len(profile.powers))],
        color=colors,
        zorder=3,
    )

    if los:
        ax.plot(
            [],
            [],
            color="#D62728",
            marker="^",
            linewidth=2.2,
            label="Componente LOS",
        )

    ax.plot(
        [],
        [],
        color="#2B1397",
        marker="^",
        linewidth=1.35,
        label="Componentes dispersas",
    )
    ax.set_yscale("log")
    ax.set_ylim(bottom=r_min, top=max_power * 1.5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("Espectro Angular de Potencia - AoA Azimutal")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.03,
        0.03,
        f"ASA = {asa:.3f} graus",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def canonicalize_polar_arrival_angles(
    azimuth_angles_deg: list[float],
    elevation_angles_deg: list[float],
) -> tuple[list[float], list[float]]:
    if len(azimuth_angles_deg) != len(elevation_angles_deg):
        raise ValueError("Listas de azimute e elevacao devem ter o mesmo tamanho.")

    azimuth_out = []
    elevation_out = []

    for phi_deg, theta_deg in zip(azimuth_angles_deg, elevation_angles_deg):
        phi = phi_deg
        theta = theta_deg

        # Mantem theta no dominio polar [0, 180] refletindo nos polos.
        # Cada reflexao polar exige giro de 180° no azimute para manter a mesma direcao 3D.
        while theta < 0.0:
            theta = -theta
            phi += 180.0
        while theta > 180.0:
            theta = 360.0 - theta
            phi += 180.0

        azimuth_out.append(wrap_angle_degrees(phi))
        elevation_out.append(theta)

    return azimuth_out, elevation_out


def calculate_multipath_unit_vectors(
    azimuth_arrival_angles_deg: list[float],
    elevation_arrival_angles_deg: list[float],
) -> list[tuple[float, float, float]]:
    if len(azimuth_arrival_angles_deg) != len(elevation_arrival_angles_deg):
        raise ValueError(
            "Listas de angulos azimutais e de elevacao devem ter o mesmo tamanho."
        )

    unit_vectors = []
    for phi_deg, theta_deg in zip(
        azimuth_arrival_angles_deg,
        elevation_arrival_angles_deg,
    ):
        phi = radians(phi_deg)
        theta = radians(theta_deg)

        # r_n = [cos(phi_n') sin(theta_n'), sin(phi_n') sin(theta_n'), cos(theta_n')]
        rx = cos(phi) * sin(theta)
        ry = sin(phi) * sin(theta)
        rz = cos(theta)
        unit_vectors.append((rx, ry, rz))

    return unit_vectors


def normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm == 0:
        raise ValueError("Vetor nao pode ser nulo para normalizacao.")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def calculate_doppler_shifts(
    fc_hz: float,
    ut_speed_m_s: float,
    ut_velocity_unit_vector: tuple[float, float, float],
    multipath_unit_vectors: list[tuple[float, float, float]],
) -> list[float]:
    c = 299_792_458.0
    wavelength = c / fc_hz
    velocity_unit = normalize_vector(ut_velocity_unit_vector)
    doppler_shifts_hz = []

    for ray_unit in multipath_unit_vectors:
        dot_product = (
            ray_unit[0] * velocity_unit[0]
            + ray_unit[1] * velocity_unit[1]
            + ray_unit[2] * velocity_unit[2]
        )
        doppler = (ut_speed_m_s / wavelength) * dot_product
        doppler_shifts_hz.append(doppler)

    return doppler_shifts_hz


def plot_doppler_spectrum(
    doppler_shifts_hz: list[float],
    magnitudes: list[float],
    los: bool,
) -> None:
    output_path = Path(__file__).with_name("doppler_spectrum_components.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    if len(doppler_shifts_hz) != len(magnitudes):
        raise ValueError("Listas de Doppler e magnitudes devem ter o mesmo tamanho.")

    positive_magnitudes = [value for value in magnitudes if value > 0]
    y_min = max(min(positive_magnitudes) / 10, max(positive_magnitudes) / 1e7)

    colors = ["#2B1397"] * len(doppler_shifts_hz)
    if los:
        colors[0] = "#D62728"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for fd, mag, color in zip(doppler_shifts_hz, magnitudes, colors):
        ax.plot([fd, fd], [y_min, mag], color=color, linewidth=1.1, alpha=0.85)

    ax.scatter(doppler_shifts_hz, magnitudes, c=colors, s=26, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=y_min, top=max(positive_magnitudes) * 1.5)
    ax.set_xlabel("Desvio Doppler (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Espectro Doppler por Componente Multipercurso")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def square_pulse(t: float, pulse_width: float) -> float:
    return 1.0 if 0.0 <= t < pulse_width else 0.0


def build_time_axis(
    delays: list[float],
    pulse_width: float,
    n_samples: int = 10000,
) -> list[float]:
    t_end = 5.0 * pulse_width
    dt = t_end / max(n_samples - 1, 1)
    return [k * dt for k in range(n_samples)]


def calculate_received_signal_from_multipath(
    fc_hz: float,
    multipath_profile: MultipathProfile,
    doppler_shifts_hz: list[float],
    pulse_width: float,
    time_axis_s: list[float],
) -> tuple[list[float], list[float]]:
    tx_signal = [square_pulse(t, pulse_width) for t in time_axis_s]
    rx_signal_magnitude = []

    for t in time_axis_s:
        real_sum = 0.0
        imag_sum = 0.0
        for alpha_n, tau_n, nu_n in zip(
            multipath_profile.magnitudes,
            multipath_profile.delays,
            doppler_shifts_hz,
        ):
            if tau_n > 5.0 * pulse_width:
                continue
            shifted_pulse = square_pulse(t - tau_n, pulse_width)
            if shifted_pulse == 0.0:
                continue

            phi_bar_n = 2 * pi * ((fc_hz + nu_n) * tau_n)
            phi_n_t = phi_bar_n - 2 * pi * nu_n * t
            real_sum += alpha_n * cos(phi_n_t) * shifted_pulse
            imag_sum -= alpha_n * sin(phi_n_t) * shifted_pulse

        rx_signal_magnitude.append(sqrt(real_sum ** 2 + imag_sum ** 2))

    return tx_signal, rx_signal_magnitude


def plot_transmitted_and_received_signal(
    time_axis_s: list[float],
    tx_signal: list[float],
    rx_signal_magnitude: list[float],
    pulse_width_s: float,
    delay_spread_s: float,
) -> None:
    output_path = Path(__file__).with_name("tx_rx_pulse_signal.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    def format_scientific(value: float) -> str:
        if value == 0:
            return "0.00 x 10^0"
        mantissa_str, exponent_str = f"{value:.2e}".split("e")
        exponent = int(exponent_str)
        return f"{float(mantissa_str):.2f} x 10^{exponent}"

    time_axis_us = [t * 1e6 for t in time_axis_s]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    ax.plot(time_axis_us, tx_signal, color="#D62728", linewidth=2.0, label="Sinal transmitido s(t)")
    ax.plot(time_axis_us, rx_signal_magnitude, color="#1F77B4", linewidth=1.7, label="Sinal recebido |r(t)|")
    ax.set_xlabel("Tempo (us)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sinal Transmitido e Sinal Recebido no Canal Multipercurso")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    pulse_width_us = pulse_width_s * 1e6
    delay_spread_us = delay_spread_s * 1e6
    pulse_bandwidth_mhz = (1.0 / pulse_width_s) / 1e6 if pulse_width_s > 0 else 0.0
    ax.text(
        0.98,
        0.78,
        (
            f"T_pulse = {format_scientific(pulse_width_us)} us\n"
            f"Banda ~ 1/T = {format_scientific(pulse_bandwidth_mhz)} MHz\n"
            f"DS = {format_scientific(delay_spread_us)} us"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def calculate_channel_autocorrelation(
    kappa_hz: float,
    sigma_s: float,
    powers: list[float],
    delays_s: list[float],
    doppler_shifts_hz: list[float],
) -> tuple[float, float]:
    if not (len(powers) == len(delays_s) == len(doppler_shifts_hz)):
        raise ValueError("Powers, delays e Doppler devem ter o mesmo tamanho.")

    omega = sum(powers)
    if omega <= 0:
        raise ValueError("Soma das potencias deve ser positiva.")

    real_sum = 0.0
    imag_sum = 0.0
    for power_n, tau_n, nu_n in zip(powers, delays_s, doppler_shifts_hz):
        phase = 2 * pi * (nu_n * sigma_s - kappa_hz * tau_n)
        real_sum += power_n * cos(phase)
        imag_sum += power_n * sin(phase)

    return real_sum / omega, imag_sum / omega


def calculate_frequency_correlation_curve(
    multipath_profile: MultipathProfile,
    doppler_shifts_hz: list[float],
    kappa_axis_hz: list[float],
) -> list[float]:
    magnitudes = []
    for kappa_hz in kappa_axis_hz:
        real_part, imag_part = calculate_channel_autocorrelation(
            kappa_hz=kappa_hz,
            sigma_s=0.0,
            powers=multipath_profile.powers,
            delays_s=multipath_profile.delays,
            doppler_shifts_hz=doppler_shifts_hz,
        )
        magnitudes.append(sqrt(real_part ** 2 + imag_part ** 2))
    return magnitudes


def calculate_time_correlation_curve(
    multipath_profile: MultipathProfile,
    doppler_shifts_hz: list[float],
    sigma_axis_s: list[float],
) -> list[float]:
    magnitudes = []
    for sigma_s in sigma_axis_s:
        real_part, imag_part = calculate_channel_autocorrelation(
            kappa_hz=0.0,
            sigma_s=sigma_s,
            powers=multipath_profile.powers,
            delays_s=multipath_profile.delays,
            doppler_shifts_hz=doppler_shifts_hz,
        )
        magnitudes.append(sqrt(real_part ** 2 + imag_part ** 2))
    return magnitudes


def plot_frequency_correlation_curve(
    kappa_axis_hz: list[float],
    rho_magnitude: list[float],
) -> None:
    output_path = Path(__file__).with_name("channel_autocorrelation_frequency.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    eps = 1e-15
    y_values = [max(value, eps) for value in rho_magnitude]
    positive_pairs = [(k, y) for k, y in zip(kappa_axis_hz, y_values) if k > 0]
    if not positive_pairs:
        raise ValueError("Eixo de frequencia deve conter valores positivos para escala log.")
    kappa_pos_hz = [pair[0] for pair in positive_pairs]
    rho_pos = [pair[1] for pair in positive_pairs]

    def find_bc(threshold: float) -> float | None:
        for kappa, value in zip(kappa_pos_hz, rho_pos):
            if value < threshold:
                return kappa
        return None

    bc_95 = find_bc(0.95)
    bc_80 = find_bc(0.80)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=140)
    ax.plot(kappa_pos_hz, rho_pos, color="#1F77B4", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Desvio de Frequencia - kappa (Hz)")
    ax.set_ylabel("|rho_TT(kappa, 0)|")
    ax.set_title("Autocorrelacao do Canal no Dominio de Frequencia")
    ax.grid(True, which="both", alpha=0.3)

    ax.axhline(0.95, color="#6E6E6E", linestyle="--", linewidth=1.2)
    ax.axhline(0.80, color="#6E6E6E", linestyle="--", linewidth=1.2)

    if bc_95 is not None:
        ax.axvline(bc_95, color="#6E6E6E", linestyle="--", linewidth=1.8)
    if bc_80 is not None:
        ax.axvline(bc_80, color="#6E6E6E", linestyle=(0, (3, 3)), linewidth=1.8)

    title_parts = []
    if bc_95 is not None:
        title_parts.append(f"Bc(rho_B=0.95)={bc_95/1e6:.2f} MHz")
    if bc_80 is not None:
        title_parts.append(f"Bc(rho_B=0.80)={bc_80/1e6:.2f} MHz")
    if title_parts:
        ax.text(
            0.98,
            0.98,
            ", ".join(title_parts),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_ylim(bottom=max(min(rho_pos), 1e-3), top=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def plot_time_correlation_curve(
    sigma_axis_s: list[float],
    rho_magnitude: list[float],
) -> None:
    output_path = Path(__file__).with_name("channel_autocorrelation_time.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    eps = 1e-15
    y_values = [max(value, eps) for value in rho_magnitude]
    positive_pairs = [(s, y) for s, y in zip(sigma_axis_s, y_values) if s > 0]
    if not positive_pairs:
        raise ValueError("Eixo de tempo deve conter valores positivos para escala log.")
    sigma_pos_s = [pair[0] for pair in positive_pairs]
    rho_pos = [pair[1] for pair in positive_pairs]

    def find_tc(threshold: float) -> float | None:
        for sigma, value in zip(sigma_pos_s, rho_pos):
            if value < threshold:
                return sigma
        return None

    tc_95 = find_tc(0.95)
    tc_80 = find_tc(0.80)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=140)
    ax.plot(sigma_pos_s, rho_pos, color="#2B1397", linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Desvio de Tempo - sigma (s)")
    ax.set_ylabel("|rho_TT(0, sigma)|")
    ax.set_title("Autocorrelacao do Canal no Dominio do Tempo")
    ax.grid(True, which="both", alpha=0.3)

    ax.axhline(0.95, color="#6E6E6E", linestyle="--", linewidth=1.2)
    ax.axhline(0.80, color="#6E6E6E", linestyle="--", linewidth=1.2)
    if tc_95 is not None:
        ax.axvline(tc_95, color="#6E6E6E", linestyle="--", linewidth=1.8)
    if tc_80 is not None:
        ax.axvline(tc_80, color="#6E6E6E", linestyle=(0, (3, 3)), linewidth=1.8)

    title_parts = []
    if tc_95 is not None:
        title_parts.append(f"Tc(rho_T=0.95)={tc_95*1e3:.2f} ms")
    if tc_80 is not None:
        title_parts.append(f"Tc(rho_T=0.80)={tc_80*1e3:.2f} ms")
    if title_parts:
        ax.text(
            0.98,
            0.98,
            ", ".join(title_parts),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_ylim(bottom=max(min(rho_pos), 1e-3), top=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")


def generate_elevation_angular_power_profile(
    profile: MultipathProfile,
    zsa: float,
    los_angle: float,
    los: bool,
) -> AngularPowerProfile:
    max_power = max(profile.powers)
    los_angle_deg = degrees(los_angle)
    elevation_arrival_angles = []

    for power in profile.powers:
        if power <= 0 or max_power <= 0:
            initial_offset = 0.0
        else:
            power_ratio = min(power / max_power, 1.0)
            initial_offset = -zsa * log(power_ratio)

        random_sign = random.choice([-1, 1])
        random_fluctuation = random.gauss(0, zsa / 7)
        angle = random_sign * initial_offset + random_fluctuation + los_angle_deg
        elevation_arrival_angles.append(angle)

    if los:
        elevation_arrival_angles[0] = los_angle_deg

    return AngularPowerProfile(arrival_angles=elevation_arrival_angles, powers=profile.powers)


def plot_elevation_angular_power_profile(
    profile: AngularPowerProfile,
    zsa: float,
    los: bool,
) -> None:
    output_path = Path(__file__).with_name("angular_power_spectrum_elevation.png")

    if plt is None:
        print("Matplotlib nao esta instalado. Instale com: pip install matplotlib")
        return

    positive_powers = [power for power in profile.powers if power > 0]
    min_power = min(positive_powers)
    max_power = max(positive_powers)
    r_min = max(min_power / 10, max_power / 1e7)
    angles_rad = [radians(angle) for angle in profile.arrival_angles]

    colors = ["#2B1397"] * len(profile.arrival_angles)
    linewidths = [1.35] * len(profile.arrival_angles)

    if los:
        colors[0] = "#D62728"
        linewidths[0] = 2.2

    fig, ax = plt.subplots(figsize=(7, 7), dpi=140, subplot_kw={"projection": "polar"})
    for angle, power, color, linewidth in zip(
        angles_rad,
        profile.powers,
        colors,
        linewidths,
    ):
        ax.plot([angle, angle], [r_min, power], color=color, linewidth=linewidth)

    ax.scatter(
        angles_rad,
        profile.powers,
        s=[42 if los and index == 0 else 24 for index in range(len(profile.powers))],
        color=colors,
        zorder=3,
    )

    if los:
        ax.plot([], [], color="#D62728", marker="o", linewidth=2.2, label="Componente LOS")
    ax.plot([], [], color="#2B1397", marker="o", linewidth=1.35, label="Componentes dispersas")

    ax.set_yscale("log")
    ax.set_ylim(bottom=r_min, top=max_power * 1.5)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Espectro Angular de Potencia - AoA Elevacao")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.04,
        f"ZSA = {zsa:.3f} graus",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Grafico salvo em: {output_path}")



def build_scenario() -> ScenarioConfig:
    d2d, d3d = calculate_distances(ut_pos, bs_pos)
    phi, phi_l, theta, theta_l = calculate_angles(ut_pos, bs_pos)
    pr_los = calculate_los_probability(scn, d2d, h_ut)

    u = random.random()
    
    #print(f"Variável Uniforme: {u*100} %")
    
    los = u <= pr_los
    large_scale_params = calculate_large_scale_parameters(scn, fc, los)
    multipath_profile = generate_multipath_profile(large_scale_params, los, n)
    azimuth_angular_power_profile = generate_azimuth_angular_power_profile(
        profile=multipath_profile,
        asa=large_scale_params["ASA"].linear_value,
        los_angle=phi_l,
        los=los,
    )
    elevation_angular_power_profile = generate_elevation_angular_power_profile(
        profile=multipath_profile,
        zsa=large_scale_params["ZSA"].linear_value,
        los_angle=theta_l,
        los=los,
    )
    azimuth_corrected, elevation_corrected = canonicalize_polar_arrival_angles(
        azimuth_angular_power_profile.arrival_angles,
        elevation_angular_power_profile.arrival_angles,
    )
    azimuth_angular_power_profile = AngularPowerProfile(
        arrival_angles=azimuth_corrected,
        powers=azimuth_angular_power_profile.powers,
    )
    elevation_angular_power_profile = AngularPowerProfile(
        arrival_angles=elevation_corrected,
        powers=elevation_angular_power_profile.powers,
    )
    multipath_unit_vectors = calculate_multipath_unit_vectors(
        azimuth_angular_power_profile.arrival_angles,
        elevation_angular_power_profile.arrival_angles,
    )
    doppler_shifts_hz = calculate_doppler_shifts(
        fc_hz=fc,
        ut_speed_m_s=ut_speed,
        ut_velocity_unit_vector=ut_vector,
        multipath_unit_vectors=multipath_unit_vectors,
    )

    return ScenarioConfig(
        los=los,
        fc=fc,
        h_bs=h_bs,
        h_ut=h_ut,
        d2d=d2d,
        d3d=d3d,
        scn=scn,
        spd=ut_speed,
        vut=ut_vector,
        theta=theta,
        phi=phi,
        theta_l=theta_l,
        phi_l=phi_l,
        pr_los=pr_los,
        large_scale_params=large_scale_params,
        multipath_profile=multipath_profile,
        azimuth_angular_power_profile=azimuth_angular_power_profile,
        elevation_angular_power_profile=elevation_angular_power_profile,
        multipath_unit_vectors=multipath_unit_vectors,
        doppler_shifts_hz=doppler_shifts_hz,
    )


def main() -> None:
    scenario = build_scenario()

    print("Parametros do cenario:")
    print(f"Cenario: {scenario.scn}")
    print(f"LOS: {'Sim' if scenario.los else 'Nao'}")
    print(f"fc: {scenario.fc:.6g} Hz")
    print(f"h_bs: {scenario.h_bs:.2f} m")
    print(f"h_ut: {scenario.h_ut:.2f} m")
    print(f"d2d calculada: {scenario.d2d:.2f} m")
    print(f"d3d calculada: {scenario.d3d:.2f} m")
    print(f"Probabilidade LOS: {scenario.pr_los:.4f}")
    print("Parametros de larga escala:")
    for name, param in scenario.large_scale_params.items():
        if param.mu is None:
            print(f"  {name}: N/A")
        elif param.sigma is None:
            print(f"  {name}: valor={param.linear_value:.6g} [{param.unit}]")
        else:
            print(
                f"  {name}: mu={param.mu:.4f}, "
                f"sigma={param.sigma:.4f} [{param.unit}], "
                f"valor_log={param.log_value:.4f}, "
                f"valor_linear={param.linear_value:.6g}"
            )
    print(f"Velocidade do UT: {scenario.spd:.2f} m/s")
    print(f"Vetor velocidade unitario do UT: {scenario.vut}")
    print(
        "Angulo de elevacao (Saida): "
        f"{scenario.theta:.2f} rad / {degrees(scenario.theta):.2f} graus"
    )
    print(
        "Angulo de elevacao (Chegada): "
        f"{scenario.theta_l:.2f} rad / {degrees(scenario.theta_l):.2f} graus"
    )
    print(
        "Angulo de azimute (Saida): "
        f"{scenario.phi:.2f} rad / {degrees(scenario.phi):.2f} graus"
    )
    print(
        "Angulo de azimute (Chegada): "
        f"{scenario.phi_l:.2f} rad / {degrees(scenario.phi_l):.2f} graus"
    )
    print(
        "Soma das potencias multipercurso: "
        f"{sum(scenario.multipath_profile.powers):.6f}"
    )
    print(f"Numero de vetores unitarios multipercurso: {len(scenario.multipath_unit_vectors)}")

    plot_multipath_profile(
        scenario.multipath_profile,
        scenario.large_scale_params["DS"].linear_value,
        scenario.los,
    )
    plot_azimuth_angular_power_profile(
        scenario.azimuth_angular_power_profile,
        scenario.large_scale_params["ASA"].linear_value,
        scenario.los,
    )
    plot_elevation_angular_power_profile(
        scenario.elevation_angular_power_profile,
        scenario.large_scale_params["ZSA"].linear_value,
        scenario.los,
    )
    plot_doppler_spectrum(
        scenario.doppler_shifts_hz,
        scenario.multipath_profile.magnitudes,
        scenario.los,
    )
    time_axis_s = build_time_axis(scenario.multipath_profile.delays, T_pulse)
    tx_signal, rx_signal_magnitude = calculate_received_signal_from_multipath(
        fc_hz=scenario.fc,
        multipath_profile=scenario.multipath_profile,
        doppler_shifts_hz=scenario.doppler_shifts_hz,
        pulse_width=T_pulse,
        time_axis_s=time_axis_s,
    )
    plot_transmitted_and_received_signal(
        time_axis_s,
        tx_signal,
        rx_signal_magnitude,
        pulse_width_s=T_pulse,
        delay_spread_s=scenario.large_scale_params["DS"].linear_value,
    )
    n_corr = 1500
    kappa_min_hz = 1.0
    kappa_max_hz = 1e10
    log_min = log10(kappa_min_hz)
    log_max = log10(kappa_max_hz)
    kappa_positive_hz = [
        10 ** (log_min + (log_max - log_min) * i / (n_corr - 1)) for i in range(n_corr)
    ]
    kappa_axis_hz = [-value for value in reversed(kappa_positive_hz)] + [0.0] + kappa_positive_hz
    sigma_min_s = 1e-6
    sigma_max_s = 1.0
    sigma_log_min = log10(sigma_min_s)
    sigma_log_max = log10(sigma_max_s)
    sigma_axis_s = [
        10 ** (sigma_log_min + (sigma_log_max - sigma_log_min) * i / (n_corr - 1))
        for i in range(n_corr)
    ]
    rho_freq_mag = calculate_frequency_correlation_curve(
        scenario.multipath_profile,
        scenario.doppler_shifts_hz,
        kappa_axis_hz,
    )
    rho_time_mag = calculate_time_correlation_curve(
        scenario.multipath_profile,
        scenario.doppler_shifts_hz,
        sigma_axis_s,
    )
    plot_frequency_correlation_curve(kappa_axis_hz, rho_freq_mag)
    plot_time_correlation_curve(sigma_axis_s, rho_time_mag)


if __name__ == "__main__":
    main()
