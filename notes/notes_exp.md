# From Non-Markovian Time Series to Koopman Autoencoders
### A step-by-step pedagogical note for Mathis

---

## 0. Roadmap

The goal of this note is to build, brick by brick, the chain of reasoning that justifies why we are looking at **Koopman autoencoders** for this project. We will go through four stages:

1. **Stage 1** — Understand *why* macro-financial time series are hard: they are generally non-Markovian, and their dependence structure is generally nonlinear.
2. **Stage 2** — Formalize discrete-time autoregressive (AR) processes, and see exactly *where* the Markov property breaks and how a simple trick (stacking) restores it — but only in the **linear** case.
3. **Stage 3** — Introduce the general idea of an **embedding** into a **latent representation**, i.e. the idea of changing coordinates so that the dynamics become simpler.
4. **Stage 4** — Introduce the **Koopman operator**, show that it is always *linear* (even for nonlinear systems), and explain why the **Koopman autoencoder** is the natural machine-learning tool to make this theory usable on real, finite, noisy, macro-financial data — turning a nonlinear AR($p$) process into a *linear* AR($1$) process in a learned latent space.

Read the sections in order: each one is a prerequisite for the next.

---

## 1. Why are macro-financial time series difficult?

Take any macro-financial series you like — industrial production, CPI inflation, a stock's realized volatility, credit spreads. Empirically, three properties show up almost systematically:

- **Persistence / memory**: the current value carries information from far in the past, beyond what a single lag can summarize (inflation persistence, volatility clustering).
- **Nonlinearity**: the way the past influences the future is not just "a weighted sum of past values" — there are thresholds, regime switches (recession vs. expansion), asymmetric reactions (volatility reacts more to negative shocks than positive ones), saturation effects.
- **Latent structure**: what we observe (a price, an index, a survey number) is usually a noisy, nonlinear function of a smaller set of "true" underlying economic/financial factors (the business cycle stage, the volatility regime, risk aversion, etc.) that we don't observe directly.

Each of these properties, on its own, breaks the tools of classical linear time-series analysis (ARMA models, Kalman filters, spectral analysis), which are built around **two working assumptions**:

> (A) the process is **Markovian** (or can be made Markovian at a reasonable, small order), and
> (B) the transition law is (at least approximately) **linear**.

The purpose of this project is to investigate a *specific and elegant way* of restoring both (A) and (B) simultaneously, using the **Koopman operator** framework, implemented practically via a **Koopman autoencoder**. Before we get there, we need to be very precise about what "Markovian" and "linear" mean for a time series, and where exactly they fail.

---

## 2. Discrete-time autoregressive (AR) processes

### 2.1 Definition

Let $(X_t)_{t \in \mathbb{Z}}$ be a real- (or vector-) valued stochastic process, $X_t \in \mathbb{R}^d$. We say $(X_t)$ is a **linear autoregressive process of order $p$**, written AR($p$), if

$$
X_t = \sum_{i=1}^{p} \Phi_i \, X_{t-i} + \varepsilon_t, \qquad \varepsilon_t \overset{\text{iid}}{\sim} \mathcal{N}(0, \Sigma),
$$

where $\Phi_1, \dots, \Phi_p \in \mathbb{R}^{d \times d}$ are fixed coefficient matrices and $(\varepsilon_t)$ is white noise independent of the past.

### 2.2 Is an AR($p$) process Markovian?

**Not in general**, and this is the crux of the issue. By definition, $(X_t)$ is Markovian (with respect to its own filtration $\mathcal{F}_t^X = \sigma(X_s : s \le t)$) if

$$
\mathbb{P}(X_{t+1} \in \cdot \mid \mathcal{F}_t^X) = \mathbb{P}(X_{t+1} \in \cdot \mid X_t).
$$

For an AR($p$) process with $p \ge 2$, this fails: knowing $X_t$ alone is *not* enough to characterize the distribution of $X_{t+1}$, because $X_{t+1}$ also depends on $X_{t-1}, \dots, X_{t-p+1}$. The process "remembers" more than just its last value — this is exactly the **memory effect** we want to deal with.

### 2.3 The stacking trick: a first (linear) embedding

There is a classical, simple fix: define the **stacked vector**

$$
Z_t := \begin{pmatrix} X_t \\ X_{t-1} \\ \vdots \\ X_{t-p+1} \end{pmatrix} \in \mathbb{R}^{dp}.
$$

Then one can check that $(Z_t)$ satisfies a **first-order**, i.e. **AR(1)**, recursion:

$$
Z_t = A \, Z_{t-1} + B \varepsilon_t, \qquad
A = \begin{pmatrix}
\Phi_1 & \Phi_2 & \cdots & \Phi_{p-1} & \Phi_p \\
I_d & 0 & \cdots & 0 & 0 \\
0 & I_d & \cdots & 0 & 0 \\
\vdots & & \ddots & & \vdots \\
0 & 0 & \cdots & I_d & 0
\end{pmatrix}.
$$

This is the **companion form**. Two things happened simultaneously:

1. $(Z_t)$ is now **Markovian**: $Z_t$ alone determines the distribution of $Z_{t+1}$.
2. $(Z_t)$ is still **linear**: the transition is a matrix multiplication.

This is the simplest possible example of a **Markovian embedding** (see the project's problem formulation): we mapped the process into a higher-dimensional space $\mathbb{R}^{dp}$ where it becomes first-order Markov.

### 2.4 Where the trick breaks: nonlinear AR($p$)

Now suppose the true dynamics are **nonlinear**:

$$
X_t = F(X_{t-1}, \dots, X_{t-p}) + \varepsilon_t,
$$

for some nonlinear function $F : (\mathbb{R}^d)^p \to \mathbb{R}^d$ (this is the realistic case for most macro-financial series: think of a threshold-AR model, a GARCH-type volatility equation, or simply an unknown nonlinear law estimated from data).

Stacking still works to restore the **Markov property**:

$$
Z_t = \begin{pmatrix} X_t \\ \vdots \\ X_{t-p+1}\end{pmatrix}, \qquad Z_t = G(Z_{t-1}) + \tilde\varepsilon_t, \quad G(Z_{t-1}) = \begin{pmatrix} F(X_{t-1}, \dots, X_{t-p}) \\ X_{t-1} \\ \vdots \\ X_{t-p+2} \end{pmatrix},
$$

so $(Z_t)$ is Markovian of order 1. **But it is no longer linear**: $G$ is a nonlinear map. We have solved problem (A) (Markovianity) but not problem (B) (linearity).

This matters a lot in practice: the entire toolbox of linear systems theory — eigenvalue/eigenvector (modal) analysis, closed-form multi-step forecasts $Z_{t+h} = A^h Z_t$, Kalman filtering, stability analysis via spectral radius, frequency-domain analysis — requires **linearity**, not just Markovianity. A nonlinear Markov chain in $\mathbb{R}^{dp}$ is progress, but it is still, in general, a black box for forecasting and interpretation.

**Question that drives the rest of this note:** *Can we find another change of coordinates — possibly into a different, cleverly chosen space — in which the dynamics become linear, exactly as stacking made them Markovian?*

---

## 3. Embeddings and latent representations

### 3.1 Generalizing "stacking"

Stacking is one specific example of a more general idea: instead of working with $X_t$ (or its raw history) directly, we construct a new variable

$$
Z_t = \phi(X_t, X_{t-1}, \dots)
$$

through some **feature map** (encoder) $\phi$, chosen so that $(Z_t)$ has nicer properties than $(X_t)$ — Markovian, linear, lower-dimensional, or all three. This new variable lives in a **latent space** $\mathcal{Z}$ (also called the *embedding space*, or *lifted space*).

For this to be useful we generally need:

- **Markovianity** (as before): $Z_{t+1} \mid \mathcal{F}_t^Z \overset{d}{=} Z_{t+1} \mid Z_t$.
- **Recoverability**: there exists a **decoder** $\psi$ such that $X_t \approx \psi(Z_t)$ — we haven't thrown away the information we actually care about.
- Ideally, **simplicity of the dynamics** in $\mathcal{Z}$: e.g., $Z_{t+1} = A Z_t$ (linear!) rather than $Z_{t+1} = G(Z_t)$ (nonlinear).

### 3.2 Two ways to obtain $\phi$

There are two broad strategies to construct such a $\phi$:

1. **Model-based / hand-crafted**: derive $\phi$ analytically from known structure (e.g. stacking, or a known state-space model where $Z_t$ is a physically meaningful hidden state).
2. **Data-driven / learned**: parametrize $\phi = \phi_\theta$ (e.g. a neural network) and *learn* it from data by optimizing a loss function that encourages the desired properties (reconstruction, predictability, linearity of the dynamics).

This second strategy is exactly what an **autoencoder** does: it learns an encoder $\phi_\theta : \mathcal{X} \to \mathcal{Z}$ and a decoder $\psi_\theta : \mathcal{Z} \to \mathcal{X}$ jointly, typically by minimizing a reconstruction loss

$$
\mathcal{L}_{\text{rec}}(\theta) = \mathbb{E} \left[ \| X_t - \psi_\theta(\phi_\theta(X_t)) \|^2 \right].
$$

A plain autoencoder only asks that $Z_t$ be a good *compressed representation* of $X_t$; it says nothing about the *dynamics* of $(Z_t)$ over time. To make the latent dynamics linear, we need one more theoretical ingredient: the **Koopman operator**. This is where the plain "embedding" idea becomes powerful enough to solve problem (B) as well as problem (A).

---

## 4. The Koopman operator: linearizing nonlinear dynamics

### 4.1 Setting

Consider a (deterministic, for now) discrete-time dynamical system on a state space $\mathcal{M}$ (think $\mathcal{M} = \mathbb{R}^{dp}$, the stacked AR($p$) state from Section 2.4):

$$
Z_{t+1} = G(Z_t), \qquad G : \mathcal{M} \to \mathcal{M} \text{ possibly nonlinear.}
$$

Instead of tracking the state $Z_t$ directly, consider **observables**: scalar (or vector) functions $g : \mathcal{M} \to \mathbb{C}$ of the state, living in some function space $\mathcal{F}$ (e.g. $\mathcal{F} = L^2(\mathcal{M}, \mu)$ for an invariant measure $\mu$, or the space of continuous bounded functions).

### 4.2 Definition of the Koopman operator

**Definition (Koopman operator).** The **Koopman operator** $\mathcal{K} : \mathcal{F} \to \mathcal{F}$ associated with the dynamics $G$ is defined by composition with $G$:

$$
(\mathcal{K} g)(z) := g(G(z)), \qquad \forall g \in \mathcal{F}, \; \forall z \in \mathcal{M}.
$$

In words: $\mathcal{K}$ takes an observable $g$ and returns the observable "$g$ one time-step later."

### 4.3 The key theorem: $\mathcal{K}$ is always linear

**Proposition.** $\mathcal{K}$ is a **linear** operator on $\mathcal{F}$, *regardless of whether $G$ is linear or nonlinear.*

**Proof.** For any $g_1, g_2 \in \mathcal{F}$ and scalars $a, b$:

$$
\mathcal{K}(a g_1 + b g_2)(z) = (a g_1 + b g_2)(G(z)) = a\, g_1(G(z)) + b\, g_2(G(z)) = a (\mathcal{K} g_1)(z) + b (\mathcal{K} g_2)(z). \qquad \blacksquare
$$

This is the entire trick, and it is worth sitting with it for a moment because it looks almost too simple: **we have not changed the dynamics** ($G$ is still nonlinear); we have **changed our point of view**, from tracking the *state* $z$ to tracking *functions of the state*. The price we pay is that $\mathcal{F}$ is (generically) **infinite-dimensional**, even though $\mathcal{M}$ was finite-dimensional. We traded nonlinearity in finite dimension for linearity in infinite dimension.

### 4.4 Spectral decomposition and Koopman modes

Because $\mathcal{K}$ is linear, it makes sense to ask about its eigenvalues and eigenfunctions:

$$
\mathcal{K} \varphi_j = \lambda_j \varphi_j, \qquad \lambda_j \in \mathbb{C}, \; \varphi_j \in \mathcal{F}.
$$

If a (vector) observable of interest $g$ can be expanded in this eigenbasis, $g = \sum_j c_j \varphi_j$, then

$$
g(Z_t) = (\mathcal{K}^t g)(Z_0) = \sum_j c_j \lambda_j^{\,t}\, \varphi_j(Z_0).
$$

This is a **closed-form, linear evolution law** for the observable, entirely analogous to the diagonalization $Z_t = A^t Z_0$ of a linear system — except here it works for a *nonlinear* $G$. The $\lambda_j$ play the role of eigenvalues of a (generally infinite) linear system: $|\lambda_j| < 1$ means a mode that decays (mean-reverts), $|\lambda_j| = 1$ a persistent/cyclical mode, $\arg(\lambda_j)$ an oscillation frequency. This is directly interpretable in macro-financial terms: business-cycle periodicity, mean-reversion speed of a spread, persistence of a shock.

### 4.5 From infinite to finite: EDMD

In practice we cannot work with an infinite-dimensional $\mathcal{F}$. **Extended Dynamic Mode Decomposition (EDMD)** approximates $\mathcal{K}$ by:

1. Choosing a finite **dictionary** of observables $g_1, \dots, g_k : \mathcal{M} \to \mathbb{R}$ (e.g. monomials, radial basis functions), stacked as $\Psi(z) = (g_1(z), \dots, g_k(z))^\top$;
2. Looking for the best finite matrix $K \in \mathbb{R}^{k \times k}$ such that $\Psi(G(z)) \approx K \Psi(z)$, estimated by least squares on data $(z_t, z_{t+1})$;
3. Using $K$ (and its eigendecomposition) as a finite-dimensional proxy for $\mathcal{K}$.

**The catch**: the quality of this approximation depends *entirely* on the choice of dictionary $\Psi$. A badly chosen dictionary gives a poor linear approximation no matter how much data you have. This is precisely the weak point that motivates the next — and final — step.

---

## 5. Koopman Autoencoders: learning the dictionary

### 5.1 The idea

Instead of *hand-picking* the dictionary $\Psi$ (monomials, RBFs, etc.), we **learn** it, using exactly the encoder/decoder machinery introduced in Section 3, but now with an extra requirement: the encoder must be chosen so that the dynamics in latent space are (approximately) **linear**, not just Markovian.

Concretely, for our nonlinear, stacked AR($p$) state $Z_t \in \mathbb{R}^{dp}$ (Section 2.4), we learn:

- An **encoder** $\phi_\theta : \mathbb{R}^{dp} \to \mathbb{R}^{k}$, mapping the (nonlinear, Markovian) stacked state into a $k$-dimensional latent code $\zeta_t := \phi_\theta(Z_t)$;
- A **linear latent evolution matrix** $A_\theta \in \mathbb{R}^{k \times k}$, such that $\zeta_{t+1} \approx A_\theta \, \zeta_t$;
- A **decoder** $\psi_\theta : \mathbb{R}^k \to \mathbb{R}^{dp}$, such that $Z_t \approx \psi_\theta(\zeta_t)$.

This is the **Koopman Autoencoder**: an autoencoder whose latent code is constrained (by construction of the loss) to evolve **linearly**.

### 5.2 The loss function

Training jointly minimizes a combination of:

$$
\mathcal{L}(\theta) = \underbrace{\mathbb{E}\|Z_t - \psi_\theta(\phi_\theta(Z_t))\|^2}_{\text{reconstruction}} \;+\; \lambda_1 \underbrace{\mathbb{E}\|\phi_\theta(Z_{t+1}) - A_\theta\, \phi_\theta(Z_t)\|^2}_{\text{linear latent dynamics (1-step)}} \;+\; \lambda_2 \underbrace{\sum_{h=1}^{H}\mathbb{E}\|Z_{t+h} - \psi_\theta\big(A_\theta^{\,h}\, \phi_\theta(Z_t)\big)\|^2}_{\text{multi-step consistency}},
$$

for hyperparameters $\lambda_1, \lambda_2 > 0$ and prediction horizon $H$. The multi-step term is important: it forces the *matrix powers* of $A_\theta$ (not just $A_\theta$ itself) to produce accurate forecasts, which is exactly what we need for genuine linear, long-horizon forecasting in latent space — this is where the eigenvalues $\lambda_j$ of $A_\theta$ (Section 4.4) become directly usable and interpretable.

### 5.3 What this buys us, precisely

Compare the three objects side by side:

| | State space | Dynamics | Markovian? | Linear? |
|---|---|---|---|---|
| Raw process $X_t$ | $\mathbb{R}^d$ | nonlinear AR($p$) | **No** | No |
| Stacked $Z_t$ (Sec. 2.3–2.4) | $\mathbb{R}^{dp}$ | $Z_t = G(Z_{t-1}) + \tilde\varepsilon_t$ | **Yes** | No (if $F$ nonlinear) |
| Koopman-autoencoder latent $\zeta_t$ | $\mathbb{R}^k$ | $\zeta_t \approx A\,\zeta_{t-1}$ | **Yes** | **Yes** |

The Koopman autoencoder is therefore best understood as a **Markovian embedding with an additional linearity constraint on the dynamics** — it is a strictly stronger requirement than the Markovian embeddings discussed in Section 1 of the main proposal (any linear Markov process is Markovian, not conversely). This is exactly the sense in which it "transforms a nonlinear AR($p$) process into a linear AR($1$) process": the AR(1)-in-$\mathbb{R}^{dp}$-but-nonlinear representation of Section 2.4 is itself re-embedded, via a learned $\phi_\theta$, into a space where the AR(1) recursion becomes **linear**.

### 5.4 Why this is attractive for macro-financial data specifically

- **Interpretability**: eigenvalues of $A_\theta$ give mean-reversion speeds and cycle lengths directly (Section 4.4) — economically meaningful quantities.
- **Long-horizon forecasting**: $A_\theta^h$ gives closed-form $h$-step-ahead forecasts, without re-simulating a nonlinear model.
- **Compatibility with linear tools**: once in latent space, the entire linear-systems / linear-filtering toolbox (Kalman filter, stability analysis, spectral analysis) becomes available again, despite the original process being nonlinear and possibly long-memory.
- **Data-driven**: unlike EDMD, we do not need to guess a good dictionary of nonlinear features by hand — the network learns it from macro-financial data directly.

---

## 6. Summary of the logical chain

$$
\underbrace{\text{Non-Markovian, nonlinear } X_t}_{\text{Section 1}}
\;\xrightarrow[\text{stacking}]{\text{Section 2}}\;
\underbrace{\text{Markovian but nonlinear } Z_t}_{\text{Section 2.4}}
\;\xrightarrow[\text{learned encoder } \phi_\theta]{\text{Sections 3–5}}\;
\underbrace{\text{Markovian AND linear } \zeta_t = \phi_\theta(Z_t)}_{\text{Koopman autoencoder}}
$$

This is the precise sense in which the Koopman autoencoder is pertinent to the project: it is the natural, learnable, finite-dimensional realization of the Koopman operator theory, and it directly targets both weaknesses (non-Markovianity, nonlinearity) identified in Sections 1–2.

---

## 7. Suggested reading order

1. **AR/state-space background**: Durbin & Koopman (2012), *Time Series Analysis by State Space Methods* — Ch. 1–4, for the stacking/companion-form idea (Section 2).
2. **Koopman operator theory**: Koopman (1931), *Hamiltonian systems and transformation in Hilbert space* (original short paper — historical, optional); then Mezić (2005), *Spectral properties of dynamical systems, model reduction and decompositions*, for the modern spectral viewpoint (Section 4).
3. **EDMD**: Williams, Kevrekidis & Rowley (2015), *A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition* (Section 4.5).
4. **Koopman Autoencoders**: Lusch, Kutz & Brunton (2018), *Deep learning for universal linear embeddings of nonlinear dynamics*, and Takeishi, Kawahara & Yairi (2017), *Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition* (Section 5) — these are the two papers to read most carefully, as they are the direct methodological basis for the model we plan to implement.

## 8. Suggested exercises (before touching real data)

1. Simulate a **linear AR(2)** process, verify numerically that the stacked $Z_t$ follows exactly $Z_t = A Z_{t-1} + B\varepsilon_t$ with the companion matrix $A$ from Section 2.3, and check the eigenvalues of $A$ against the AR(2) stability condition you know from your econometrics course.
2. Simulate a **nonlinear** AR(2) (e.g. a threshold-AR or a logistic map with noise), stack it into $Z_t$, and verify empirically that $Z_t$ is Markovian (condition future values on $Z_t$ vs. on the full history and compare) but that no single matrix $A$ fits $Z_{t+1} \approx A Z_t$ well.
3. On the same simulated nonlinear series, implement a toy **EDMD** with a small polynomial dictionary, and observe how forecast quality depends on the choice of dictionary — this motivates why we want to *learn* it instead (bridge to Section 5).
4. Only after (1)–(3): implement a minimal Koopman autoencoder (encoder/decoder + linear layer $A_\theta$) on the same toy nonlinear series, and compare its multi-step forecasts to the hand-crafted EDMD from (3).