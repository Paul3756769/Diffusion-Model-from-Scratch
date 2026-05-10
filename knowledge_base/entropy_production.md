# Stochastic Thermodynamics of Diffusion

## Abstract
- Goal: Present a concise mathematical formulation of entropy production and
	information erasure for diffusion-based generative models.

## 1. Micro–Macro Entropy Connection

Stochastic (trajectory-wise) entropy (surprisal) is defined as
$$s(x,t) = -\ln p(x,t).$$

The macroscopic (Shannon) entropy is the ensemble average
$$S(t) = \langle s(x,t)\rangle = -\int p(x,t)\,\ln p(x,t)\,dx.$$

This relation links the uncertainty of a single noisy sample to the total
information content of the distribution at time $t$.

## 2. Langevin Dynamics and Harmonic Potential

We model the forward diffusion as a Langevin SDE (Ornstein–Uhlenbeck type):
$$
dX_t = f(X_t)\,dt + \sqrt{2D} \,dW_t.
$$ 

A common parameterization uses a time-dependent rate $\beta(t)$ with
$$f(X_t) = -\tfrac{1}{2}\beta(t)\,X_t,\qquad D = \tfrac{1}{2}\beta(t).$$

The drift is (effectively) conservative and can be associated with the
quadratic potential
$$V(X)=\tfrac{1}{4}X^2,\qquad f(X) = -\nabla V(X)\quad\text{(up to constants).}$$

For these linear drift and diffusion choices the process is Ornstein–Uhlenbeck,
with Gaussian marginals and analytically tractable path integrals.

## 3. Entropy Production Balance (Trajectory Level)

The second law for stochastic thermodynamics at the trajectory level reads

$$
\Delta s_{\mathrm{tot}} = \Delta s_{\mathrm{sys}} + \Delta s_{\mathrm{env}} \ge 0.
$$ 

System entropy change between times $0$ and $T$ is
$$
\Delta s_{\mathrm{sys}} = s(X_T,T)-s(X_0,0) = -\ln p(X_T,T) + \ln p(X_0,0).
$$

Environmental entropy represents the entropy/heat exchanged with the noise
bath. Using Stratonovich integration ($\circ$) and the conservative harmonic drift,
one obtains the closed form
$$\Delta s_{\mathrm{env}} = \frac{1}{D}\int f(X)\circ dX = -\int X\circ dX = \tfrac{1}{2}(X_0^2 - X_T^2).$$

Thus the trajectory-wise total entropy production simplifies to
$$\Delta s_{\mathrm{tot}} = -\ln p(X_T,T) + \ln p(X_0,0) + \tfrac{1}{2}(X_0^2 - X_T^2).$$

Taking expectations yields the usual nonnegativity of average entropy
production.

## 4. Physical Interpretation

- Irreversibility: Nonnegativity of $\Delta s_{\mathrm{tot}}$ quantifies the
	irreversibility of the forward diffusion.
- Information erasure: The term $\tfrac{1}{2}(X_0^2 - X_T^2)$ measures the
	"cost" of transforming structured data (typically larger $|X_0|$) into
	noise (smaller $|X_T|$).
- Generative modeling view: Training a reverse process aims to reduce surprisal
	$s(x,t)$ of generated samples and recover high-probability configurations
	from noise, effectively undoing part of the environmental entropy increase.

## Notes and Conventions

- Calculus: Stratonovich integrals ($\circ$) are used so standard chain rules
  apply.
- Model assumptions: Linear drift (OU) and quadratic potential are idealized
	for analytic clarity; nonlinear drifts require generalized path-integral
	expressions and do not yield the simple closed form above.
- Keywords: Stratonovich calculus, Ornstein–Uhlenbeck process, Shannon entropy,
	information erasure.

---

Advanced technical summary: trajectory-wise surprisal $s(x,t)$, Shannon entropy
recovery via averaging, OU dynamics for forward diffusion, and a closed-form
expression for environmental entropy in the harmonic case.


