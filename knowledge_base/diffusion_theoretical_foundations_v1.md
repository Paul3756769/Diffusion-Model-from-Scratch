# Theoretical Foundations of Generative Diffusion Models: 1D Langevin & SDEs

This document serves as the consolidated mathematical source of truth for building continuous-time diffusion models, rigorously defining the forward processes, time-reversal theorems, score functions, and numerical discretizations.

## 1. Stochastic Differential Equations & Langevin Dynamics

The evolution of a system subject to deterministic forces and stochastic noise is modeled via an Itô Stochastic Differential Equation (SDE):

$$dx_t = f(x_t, t)dt + g(t)dW_t$$

* $f(x_t, t)$: Drift coefficient (deterministic vector field).
* $g(t)$: Diffusion coefficient (controls the scale of the variance).
* $dW_t$: Standard Wiener process (Brownian motion) increment.

**Overdamped Langevin Equation:**
In the high-friction limit, particle momentum dissipates instantly, yielding the overdamped Langevin equation:

$$dx_t = -\nabla U(x_t)dt + \sqrt{2D}dW_t$$

where $U(x_t)$ is the potential field and $D$ is the diffusion coefficient.

## 2. The Forward Process: 1D Ornstein-Uhlenbeck (OU) Process

Generative diffusion models destroy data by pulling it toward the origin while injecting noise. This is achieved using the Ornstein-Uhlenbeck (OU) process. Let the noise schedule be denoted by $\beta(t)$. Setting the drift to a harmonic potential $U(x) = \frac{1}{4}\beta(t)x^2$ and $2D = \beta(t)$, we obtain the forward SDE:

$$dx_t = -\frac{1}{2}\beta(t) x_t dt + \sqrt{\beta(t)} dW_t$$

For a constant schedule $\beta(t) = \beta_0$ or $\lambda = \frac{1}{2}\beta_0$, the generic 1D SDE is:

$$dx_t = -\lambda x_t dt + \sqrt{2D} dW_t$$

## 3. Analytical Solution and Moments

To solve $dx_t = -\lambda x_t dt + \sqrt{2D} dW_t$, multiply by the integrating factor $e^{\lambda t}$:

$$d(x_t e^{\lambda t}) = e^{\lambda t} dx_t + \lambda e^{\lambda t} x_t dt = e^{\lambda t} \sqrt{2D} dW_t$$

Integrating from $0$ to $t$ yields the formal solution:

$$x_t = x_0 e^{-\lambda t} + \sqrt{2D} \int_0^t e^{-\lambda(t - s)} dW_s$$

### Moments
Taking the expectation $\mathbb{E}[\cdot]$, and utilizing the fact that the Itô integral of a deterministic function with respect to Brownian motion has zero mean and isometry:

* **Mean:** $\mu(t) = \mathbb{E}[x_t] = x_0 e^{-\lambda t}$
* **Variance:** $\sigma^2(t) = \text{Var}(x_t) = 2D \int_0^t e^{-2\lambda(t-s)} ds = \frac{D}{\lambda}(1 - e^{-2\lambda t})$

Because $x_t$ is a linear combination of Gaussian increments, the transition probability kernel $p(x_t | x_0)$ is strictly Gaussian:

$$p(x_t | x_0) = \mathcal{N}\left(x_t ; x_0 e^{-\lambda t}, \frac{D}{\lambda}(1 - e^{-2\lambda t})\right)$$

*Note: In standard Variance-Preserving (VP) diffusion with $2D = \beta(t)$ and $\lambda = \frac{1}{2}\beta(t)$, this yields $\sigma^2(t) = 1 - e^{-\int_0^t \beta(s)ds}$, ensuring the stationary distribution converges to $\mathcal{N}(0, 1)$ as $t \to \infty$.*

## 4. Time Reversal and Anderson's Theorem

**Fokker-Planck Equation (FPE):**
The forward SDE induces a time-evolution of the probability density $p(x,t)$ governed by the FPE:

$$\frac{\partial p}{\partial t} = -\nabla_x \cdot (f p) + \frac{1}{2} g^2 \nabla_x^2 p$$

**Anderson's Theorem (1982):**
The time-reversal of a diffusion process is also a diffusion process. If the forward process runs from $t = 0$ to $T$, the exact backward process evolving backwards in time (where $d\bar{t}$ is a negative time increment) is given by:

$$dx_t = \left[ f(x_t, t) - g(t)^2 \nabla_x \log p(x_t, t) \right] dt + g(t) d\bar{W}_t$$

* $d\bar{W}_t$: Standard Brownian motion flowing backward in time.
* $\nabla_x \log p(x_t, t)$: The **Stein Score Function**, a vector field pointing toward higher probability density.

The term $-g(t)^2 \nabla_x \log p(x_t, t)$ acts as an active corrective force, counteracting the entropic dispersion of the forward drift.

## 5. Analytical Score Functions

The score function $\nabla_x \log p(x,t)$ can be derived analytically for specific initial distributions.

### Case 1: Point Mass (Delta Distribution)
If the data is a single point $x_0$, the initial distribution is $p(x,0) = \delta(x - x_0)$. The marginal distribution at time $t$ is the transition kernel $p(x_t|x_0) = \mathcal{N}(\mu(t), \sigma^2(t))$.

The log-density is:
$$\log p(x,t) = -\frac{1}{2}\log(2\pi\sigma^2(t)) - \frac{(x - \mu(t))^2}{2\sigma^2(t)}$$

The analytical score is the exact gradient:
$$\nabla_x \log p(x,t) = -\frac{x - \mu(t)}{\sigma^2(t)} = -\frac{x - x_0 e^{-\lambda t}}{\frac{D}{\lambda}(1 - e^{-2\lambda t})}$$

### Case 2: Gaussian Mixture (Bimodal)
If the data is a superposition of two points $\{-a, a\}$, the initial distribution is $p(x,0) = \frac{1}{2}\delta(x+a) + \frac{1}{2}\delta(x-a)$.

The marginal distribution is a mixture of Gaussians with $m_t = ae^{-\lambda t}$ and $v_t = \sigma^2(t)$:
$$p(x,t) = \frac{1}{2\sqrt{2\pi v_t}} \left[ \exp\left(-\frac{(x+m_t)^2}{2v_t}\right) + \exp\left(-\frac{(x-m_t)^2}{2v_t}\right) \right]$$

Let $E_+ = \exp\left(-\frac{(x+m_t)^2}{2v_t}\right)$ and $E_- = \exp\left(-\frac{(x-m_t)^2}{2v_t}\right)$.
Taking the derivative $\partial_x p(x,t)$ and dividing by $p(x,t)$ yields:

$$\nabla_x \log p(x,t) = -\frac{x}{v_t} - \frac{m_t}{v_t} \left( \frac{E_+ - E_-}{E_+ + E_-} \right)$$

Using the identity $\frac{e^{-A} - e^{A}}{e^{-A} + e^{A}} = -\tanh(A)$ where $A = \frac{x m_t}{v_t}$, the exact score is:

$$\nabla_x \log p(x,t) = -\frac{x}{v_t} + \frac{m_t}{v_t} \tanh\left(\frac{x m_t}{v_t}\right)$$

## 6. Numerical Discretization and Sampling

### Forward Discretization (Euler-Maruyama)
To simulate the forward SDE $dx_t = -\frac{1}{2}\beta(t) x_t dt + \sqrt{\beta(t)} dW_t$ with finite step $\Delta t$:

$$x_{i+1} = x_i - \frac{1}{2}\beta(t_i)x_i \Delta t + \sqrt{\beta(t_i)\Delta t} \epsilon_i$$
where $\epsilon_i \sim \mathcal{N}(0, 1)$.

### Direct Forward Sampling
Because the OU process transition kernel is Gaussian, we can directly sample $x_t$ from $x_0$ in a single step without Euler integration. Let $\bar{\alpha}_t = \exp\left(-\int_0^t \beta(s)ds\right) \approx \prod_{i=1}^t (1 - \beta_i)$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

### Reverse Discretization
Using a neural network to estimate the score $s_\theta(x_t, t) \approx \nabla_x \log p_t(x_t)$, the backward SDE is solved via reverse Euler-Maruyama:

$$x_{t-\Delta t} = x_t - \left[ f(x_t, t) - g(t)^2 s_\theta(x_t, t) \right] \Delta t + g(t) \sqrt{\Delta t} z$$
where $z \sim \mathcal{N}(0, 1)$.

## 7. Thermodynamics and Entropy Production

For a single point $x_t$ moving through the diffusion process, the total entropy produced consists of two parts:

**1. System Entropy Change ($\Delta s_{sys}$):** The change in the microscopic surprisal $s(x, t) = -\log p(x,t)$.
$$\Delta s_{sys} = -\log p(x_T, T) + \log p(x_0, 0)$$

**2. Environmental Entropy Change ($\Delta s_{env}$):** The information dissipated into the noise bath. For the OU process:
$$\Delta s_{env} = \frac{1}{D} \int_0^T f(x_t) dx_t = \frac{x_0^2 - x_T^2}{2}$$
