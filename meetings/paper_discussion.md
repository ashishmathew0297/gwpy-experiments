# The Rayleigh Statistic Paper

Link: https://arxiv.org/pdf/2403.12731

This paper proposes an extension of the Rayleigh statistic to evaluate the Gaussianity in strain data and identify non-Gaussian noise. The data used as well as Monte Carlo simulations for the background are based on strain data from KAGRA.

##### Null Hypothesis ($H_0$):

The null hypothesis is that the noise is Gaussian.

##### Alternative Hypothesis ($H_1$):

The alternative hypothesis is that the noise is non-Gaussian.

### Inputs to the Statistical Test

It uses the time series strain data $x(t)$ at a resolution $T$, splits it into segments $T_{FFT}$ with an overlap of $T_{FFT}/2$, and the sample size being computed as $n = 1 + \left\lfloor \frac{T - T_{FFT}}{T_{FFT} - \frac{T_{FFT}}{2}} \right\rfloor$ ($\lfloor\rfloor$ indicate rounding down)

The one sided PSD for each of these segments is computed as

$$P_i(f) = \frac{2|X_i(f)|^2}{T_{FFT}}$$

Where $X_i(f)$ is the frequency spectrum computed using the FFT. The $P_i(f)$ values for each segment and the mean $\bar{P(f)} = \frac{1}{n}\sum\limits_i^n P_i(f)$ are used to compute the statistic.

### How the p-value is computed

Simulated samples of background distribution (White Gaussian Noise) are computed using the Monte Carlo method. The Rayleigh statistic $R$ is computed for this simulated data with sample size $n$ at a sampling rate of 4096 Hz. $R$ is computed between 100 to 1900 Hz.

The p-value is computed by comparing the $R$ statistic for the sample noise and the samples of the background distribution with the following formula

$$
p = \cases{\frac{2}{n_{bg}}\sum\limits_{i=1}^{n_{bg}} \mathbb{1}_{\{R \geq R\}}\quad\{R \geq R_m\}\\
\frac{2}{n_{bg}}\sum\limits_{i=1}^{n_{bg}} \mathbb{1}_{\{R < R_m\}}\quad\{R < R_m\}}
$$

Here
- $R_i$ is the $i$-th value of sorted samples of background distribution
- $R_m$ is the sample median of background distribution
- $n_{bg}$ is the number of samples in the background distribution
- $\mathbb{1}_{\{\}}$ is a function whose value is $1$ when the condition in $\{\}$ is satisfied and $0$ otherwise

# Andrew's Paper

Link: https://arxiv.org/pdf/2306.09019

This paper presents a statistical test to check for excess non-Gaussian noise in GW data. This is done with the help of Bayesian Statistical Modelling. 

### Inputs to the ~~Statistical Test~~ Bayesian Model

The Q-transform of the data is computed and the Q-tiles are normalized to allow making an assumption about Gaussian data. This is used as the inputs for the Bayesian Statistical model.

Data from LIGO Livingston during the third observing run (particularly GW200129 and scattered light noise from Jan 6, 2020) was used in this case.

### How the p-value is computed

Bayesian modelling does not use p-values, hence there are no p-value computations as such.

In this paper, posterior probabilities of the time series are to model the non-Gaussian noise in the GW data by assuming the Gaussian part to be a $\chi^2$ distribution and the non-Gaussian part to be an unknown distribution. Data with excess power would have higher energies and hence a longer tail in comparison to Gaussian data.

# The Q-Gram Paper (Derek Davis)

Link: https://arxiv.org/pdf/2208.12338

This paper introduces a modification of the Q-transform called the Q-gram to rapidly compute a p-value and measure the significance of excess power in time series data.

##### Null Hypothesis ($H_0$):
The null hypothesis is that the distribution of energy tiles in the Q-gram is consistent with the distribution expected from Gaussian noise.
### Inputs to the Q-gram

The Q-gram is tested with
- Simulated GW signals injected on a Gaussian background
- Simulated GW signals injected near a glitch
- Real GW signals from LIGO during O1, O2, O3


### How the p-value is computed
Each energy tile of the spectrogram is assigned a significance.

For stationary white noise, the energies are expected to follow an exponential distribution given by

$$
P(\epsilon > \epsilon[m,k] \propto \exp{(-\epsilon[m,k])/\mu_k})
$$

Here
- $k$ is the specific frequency
- $\mu_k$ is the mean tile energy at the given frequency
- $m$ is related to the time of the event from discretizing the Q-transform

By using this to assign a probability/significance to each tile, the energy distributions of the tiles can be fit to it with a little bit of deviation from the prediction with the following formula,

$$
P(\epsilon) = Ae^{-\epsilon t}
$$

where
- $\epsilon$ is the tile energy
- $A$ and $t$ are fit parameters

To calculate the significance, the p-value for the null hypothesis (given above) is computed.
- First, for each tile, the probability of the Q-gram containing tiles with energies above the given tile is found.
	- Using the Q-gram size $N$ and computed $P(\epsilon)$ value for each of the tiles, this is found by multiplying the size of the Q-gram with the integral of probabilities that the energy is greater than equal to the magnitude of the given tile $\epsilon_0$ (aka threshold energy).

$$
\begin{align}
	\tilde{P}(\epsilon > \epsilon_0) &= \int\limits_{\epsilon_0}^{\infty}
	P(\epsilon')d\epsilon' \notag \\
    &= \int\limits_{\epsilon_0}^{\infty} \frac{Ae^{-\epsilon't}}{N} d\epsilon' \notag \\
    &= \frac{Ae^{-\epsilon_0t}}{tN}
\end{align}
$$

- After this, the probability of the whole Q-gram is found using the same Gaussian distribution assumption, taking the mean as the expected number of tiles.
	- The tile energies are assumed to follow a Poisson process with the same null hypothesis assumption. The probability of at least one tile being greater than a given tile's energy $\epsilon_0$ is based on the number of tiles above the given energy level $\lambda = \tilde{P}(\epsilon > \epsilon_0)$ and the total number of tiles $\tau = N$

$$
\begin{align}
   P(\text{Q-gram} \max \epsilon = \epsilon_0) &= 1 - e^{-\lambda \tau} \notag \\
    &= 1 - \exp \left[ - \cancel{N}\frac{Ae^{-\epsilon_0 t}}{t\cancel{N}} (N) \right] \notag\\
    &= 1 - \exp \left[ - \frac{ANe^{-\epsilon_0 t}}{t} \right]
\end{align}
$$

To determine the presence of a glitch a p-value threshold is set ($\epsilon_0$) and the tile energy probabilities are computed. Tiles with energies above this threshold are said to be containing non-Gaussian feautres, pointing to the presence of glitches.