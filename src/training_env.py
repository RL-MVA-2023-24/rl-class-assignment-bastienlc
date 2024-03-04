import numpy as np
from numba import njit

from interface import Agent

T1Upper = 1e6
T1starUpper = 5e4
T2Upper = 3200.0
T2starUpper = 80.0
VUpper = 2.5e5
EUpper = 353200.0
upper = np.array(
    [
        T1Upper,
        T1starUpper,
        T2Upper,
        T2starUpper,
        VUpper,
        EUpper,
    ]
)
T1Lower = 0.0
T1starLower = 0.0
T2Lower = 0.0
T2starLower = 0.0
VLower = 0.0
ELower = 0.0
lower = np.array(
    [
        T1Lower,
        T1starLower,
        T2Lower,
        T2starLower,
        VLower,
        ELower,
    ]
)
actions = [np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]]


def rawstate(T1, T1star, T2, T2star, V, E):
    return np.array([T1, T1star, T2, T2star, V, E])


def f_state(s, clipping=True, logscale=False):
    if clipping:
        np.clip(s, lower, upper, out=s)
    if logscale:
        s = np.log10(s)
    return s


@njit
def _reset_patient_parameters(domain_randomization=False):
    if domain_randomization:
        k1 = np.random.uniform(low=5e-7, high=8e-7)
        k2 = np.random.uniform(low=0.1e-4, high=1.0e-4)
        f = np.random.uniform(low=0.29, high=0.34)
    else:
        k1 = 8e-7
        k2 = 1e-4
        f = 0.34
    lambda1 = 1e4
    d1 = 1e-2
    m1 = 1e-5
    rho1 = 1
    lambda2 = 31.98
    d2 = 1e-2
    m2 = 1e-5
    rho2 = 1
    delta = 0.7
    NT = 100
    c = 13
    lambdaE = 1
    bE = 0.3
    Kb = 100
    dE = 0.25
    Kd = 500
    deltaE = 0.1

    return (
        k1,
        k2,
        f,
        lambda1,
        d1,
        m1,
        rho1,
        lambda2,
        d2,
        m2,
        rho2,
        delta,
        NT,
        c,
        lambdaE,
        bE,
        Kb,
        dE,
        Kd,
        deltaE,
    )


@njit
def der(state, action, *params):
    T1, T1star, T2, T2star, V, E = state
    (
        k1,
        k2,
        f,
        lambda1,
        d1,
        m1,
        rho1,
        lambda2,
        d2,
        m2,
        rho2,
        delta,
        NT,
        c,
        lambdaE,
        bE,
        Kb,
        dE,
        Kd,
        deltaE,
    ) = params

    eps1, eps2 = action

    T1dot = lambda1 - d1 * T1 - k1 * (1 - eps1) * V * T1
    T1stardot = k1 * (1 - eps1) * V * T1 - delta * T1star - m1 * E * T1star
    T2dot = lambda2 - d2 * T2 - k2 * (1 - f * eps1) * V * T2
    T2stardot = k2 * (1 - f * eps1) * V * T2 - delta * T2star - m2 * E * T2star
    Vdot = (
        NT * delta * (1 - eps2) * (T1star + T2star)
        - c * V
        - ((rho1 * k1 * (1 - eps1) * T1 + rho2 * k2 * (1 - f * eps1) * T2) * V)
    )
    Edot = (
        lambdaE
        + bE * (T1star + T2star) * E / (T1star + T2star + Kb)
        - dE * (T1star + T2star) * E / (T1star + T2star + Kd)
        - deltaE * E
    )

    return np.array([T1dot, T1stardot, T2dot, T2stardot, Vdot, Edot])


@njit
def transition(state, action, duration, *params):
    state0 = np.copy(state)
    nb_steps = int(duration // 1e-3)
    for _ in range(nb_steps):
        der_vals = der(state0, action, *params)
        state1 = state0 + der_vals * 1e-3
        state0 = state1
    return state1


@njit
def reward(state, action, state2, Q, R1, R2, S):
    return -(Q * state[4] + R1 * action[0] ** 2 + R2 * action[1] ** 2 - S * state[5])


def step(
    state,
    action,
    params,
    clipping=True,
    logscale=False,
):
    action = actions[action]

    state2 = transition(state, action, 5, *params)
    rew = reward(state, action, state2, 0.1, 20000.0, 20000.0, 1000.0)

    state2 = f_state(state2, clipping=clipping, logscale=logscale)

    return state2, rew, False, False, {}


def init(domain_randomization=False):
    return f_state(
        rawstate(T1=163573.0, T1star=11945.0, T2=5.0, T2star=46.0, V=63919.0, E=24.0),
        clipping=True,
        logscale=False,
    ), _reset_patient_parameters(domain_randomization=domain_randomization)


def evaluate_agent(agent: Agent, domain_randomization: bool = False) -> float:
    obs, params = init(domain_randomization=domain_randomization)
    episode_reward = 0
    for _ in range(200):
        action = agent.act(obs)
        obs, reward, _, _, _ = step(obs, action, params)
        episode_reward += reward
    return episode_reward
