import numpy as np
import matplotlib.pyplot as plt

# Simulate a "stock price" using a random walk
np.random.seed(42)
n = 1000
price = np.cumsum(np.random.randn(n)) + 100

# Function to calculate Hurst Exponent
def hurst_exponent(ts):
    lags = range(2, 100)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

hurst = hurst_exponent(price)
print(f"Hurst Exponent: {hurst:.3f}")

# Plot
plt.figure(figsize=(10, 4))
plt.plot(price, label='Simulated Price')
plt.title("Random Walk (Simulated Stock Price)\nHurst Exponent ≈ {:.3f}".format(hurst))
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()