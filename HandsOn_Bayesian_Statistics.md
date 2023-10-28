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
