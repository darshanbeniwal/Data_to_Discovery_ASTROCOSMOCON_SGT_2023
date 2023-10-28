# 1. Install the required packages
 
```python  
!pip install corner
```
# 2. Import required packages

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
import corner
```
# 3. Load Mock data from GitHub

```python
z,hz,hzerr=np.loadtxt("https://raw.githubusercontent.com/darshanbeniwal/Data_to_Discovery_ASTROCOSMOCON_SGT_2023/main/Hubble_30.txt",unpack=True)
```
# 4. Define Likelihood Function

```python
 def likelihood(theta, z,hz,hzerr):
    h0, om= theta
    model = h0*np.sqrt(om*(1+z)**3+1-om)
    return (np.sum(-0.5*((hz-model)/hzerr)**2-0.5*np.log(2*np.pi*hzerr**2)))
```
# 5. Define Prior Function

```python
 def prior(theta):
    h0, om= theta
    if 50.0< h0 < 80 and 0.15 < om < 0.55:
        return np.log10(1.0 / ((80 - 50.0) * (0.55 - 0.15)))
    return -np.inf
```
# 6. Define Posterior Function

```python
 def posterior(theta, z,hz,hzerr):
    lp = prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + likelihood(theta, z,hz,hzerr))
```
# 7. Define Metropolis-Hastings Algorithm Function

```python
def Metropolis_Hastings(parameter_init, nsteps):
    result = []  # List to store the sampled parameter values
    result.append(parameter_init)  # Add the initial parameter values to the result list
    for t in range(nsteps):  # Iterate over the specified number of steps
        step_var = [1, 0.1]  # Variance of the proposal distribution for each parameter
        proposal = norm.rvs(loc=result[-1], scale=step_var)  # Generate a proposal parameter value from a normal distribution
        probability = np.exp(posterior(proposal,z,hz,hzerr) - posterior(result[-1],z,hz,hzerr))  # Calculate the acceptance probability
        if (uniform.rvs() < probability):  # Accept the proposal with the acceptance probability
            result.append(proposal)  # Add the proposal to the result list
        else:
            result.append(result[-1])  # Reject the proposal and add the previous parameter value to the result list
    return(result)  # Return the sampled parameter values
```
# 8. Define Initial Seeds, Number of Steps

```python
 
```
# 9. Run Metropolis-Hastings Algorithm

```python
 
```
# 10. Plot the chains

```python
 
```
# 11. Remove Burn-in Phase

```python


```

```python

```
# 12. Replot the Chains

```python


```
# 13. Find Best Fit value of parameters

```python

```
# 14. Show Parameter Histograms

```python

```
# 15. Plot Contour

```python

```
