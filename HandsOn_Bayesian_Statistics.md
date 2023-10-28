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
h0_ini,om_ini=60,0.25
initials=h0_ini,om_ini
ndim=2
nsteps=100000
```
# 9. Run Metropolis-Hastings Algorithm

```python
result = Metropolis_Hastings(initials, nsteps)
samples_MH=np.array(result)
#will take around 26 sec
```
# 10. Plot the chains

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
samples = samples_MH.T

# Plot the traceplot of H0
axes[0].plot(samples[0], "g")
axes[0].set_ylabel("$H_0$")

# Plot the traceplot of Om
axes[1].plot(samples[1], "r")
axes[1].set_ylabel("$\Omega_{m0}$")
```
# 11. Remove Burn-in Phase

```python

nburn_in=1000
result_b = result[nburn_in:]
samples_MH_b=np.array(result_b)
```

```python

```
# 12. Replot the Chains

```python
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
samples_b = samples_MH_b.T

# Plot the traceplot of H0
axes[0].plot(samples_b[0], "g")
axes[0].set_ylabel("$H_0$")

# Plot the traceplot of Om
axes[1].plot(samples_b[1], "r")
axes[1].set_ylabel("$\Omega_{m0}$")


```
# 13. Find Best Fit value of parameters

```python
h0_chain=samples_MH_b[:,0]
om_chain=samples_MH_b[:,1]
#Estimate the mean of a and b chains
h0_best = np.mean(h0_chain)
om_best = np.mean(om_chain)

#Estimate the Std. Deviation of a and b chains

sig_h0 = np.std(h0_chain)
sig_om = np.std(om_chain)

print("Best fit values:")
print("H0:",h0_best, "Sig_h0:", sig_h0)
print("Om:",om_best, "Sig_om:", sig_om)
```
# 14. Show Parameter Histograms

```python
plt.figure(figsize=(8, 3)) #Plot Size

# Plot the histogram of a
plt.subplot(1, 2, 1)
plt.hist(h0_chain, bins=100, color='blue')
plt.xlabel('H0')
plt.ylabel('Count')

# Plot the histogram of b
plt.subplot(1, 2, 2)
plt.hist(om_chain, bins=100, color='blue')
plt.xlabel('Om')
plt.ylabel('Count')
```
# 15. Plot Contour

```python
fig = corner.corner(samples_MH,bins=40,color="b",labels=['$H_0$','$\Omega_{m0}$'],truths=[h0_best,om_best],fill_contours=True,
                    levels=(0.68,0.95,0.99,),
                    smooth=True,
                    quantiles=[0.16, 0.5, 0.84],title_fmt='.3f',plot_datapoints=False,show_titles=True)
```
