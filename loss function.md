

### 1. The Forward Process and the Transition Kernel

In the forward diffusion process (often modeled as an Ornstein-Uhlenbeck process), we incrementally inject thermal noise into our initial data state, $x_0$. 

At any time $t$, the noisy state $x_t$ is a linear combination of the original signal and standard Gaussian noise:
$$x_t = \alpha_t x_0 + \sigma_t \epsilon$$

Where:
*   $x_0$ is the uncorrupted data point.
*   $\alpha_t$ dictates the signal retention (how much of $x_0$ remains).
*   $\sigma_t$ is the standard deviation of the noise at time $t$, representing the thermal variance of the system.
*   $\epsilon \sim \mathcal{N}(0, I)$ is pure, unit-variance Gaussian noise. It represents the random "kick" applied to the particle in the heat bath.

Because the transition from $x_0$ to $x_t$ is purely linear and driven by Gaussian noise, the conditional probability distribution (the transition kernel) is exactly Gaussian:
$$q(x_t | x_0) = \mathcal{N}(x_t; \alpha_t x_0, \sigma_t^2 I)$$

### 2. Deriving the Analytical Score Function

The "score" of a probability distribution is a vector field pointing in the direction of the steepest ascent of log-probability. In statistical mechanics, this is directly analogous to the thermodynamic restoring force pulling a system toward its equilibrium (highest entropy) state.

To find the score of our conditional transition kernel, we first write out its probability density function:
$$q(x_t | x_0) = \frac{1}{(2\pi\sigma_t^2)^{D/2}} \exp\left( - \frac{\| x_t - \alpha_t x_0 \|^2}{2\sigma_t^2} \right)$$

Next, we take the natural logarithm. The normalization constant becomes a negligible additive constant ($C$), and the exponential cancels out:
$$\log q(x_t | x_0) = - \frac{\| x_t - \alpha_t x_0 \|^2}{2\sigma_t^2} + C$$

The score is defined as the gradient ($\nabla_{x_t}$) of this log-probability. Taking the derivative with respect to $x_t$ yields:
$$\nabla_{x_t} \log q(x_t | x_0) = - \frac{x_t - \alpha_t x_0}{\sigma_t^2}$$

We know from our physical equation that the noise component is $x_t - \alpha_t x_0 = \sigma_t \epsilon$. Substituting this back into the gradient:
$$\nabla_{x_t} \log q(x_t | x_0) = - \frac{\sigma_t \epsilon}{\sigma_t^2}$$

Simplifying the fraction gives us the exact analytical score function:
$$\nabla_{x_t} \log q(x_t | x_0) = - \frac{\epsilon}{\sigma_t}$$

Physical Intuition: The vector $\epsilon$ pushed the particle away from its mean. The score function naturally points in the exact opposite direction ($-\epsilon$), scaled inversely by the current variance ($\sigma_t$). It is the exact vector needed to reverse the random walk.

### 3. The General Mean Squared Error (Fisher Divergence)

Our goal is to train a neural network, $s_\theta(x_t, t)$, to approximate the true marginal score of the data, $\nabla_{x_t} \log q(x_t)$. The general objective for this is minimizing the Fisher Divergence—a Mean Squared Error (MSE) between the network's prediction and the true score:
$$\mathcal{J}(\theta) = \mathbb{E}{q(x_t)} \left[ \left\| s\theta(x_t, t) - \nabla_{x_t} \log …
