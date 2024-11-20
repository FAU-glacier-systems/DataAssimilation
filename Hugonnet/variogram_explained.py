import gstools as gs
import matplotlib.pyplot as plt

# structured field with a size 100x100 and a grid-size of 1x1
x = y = range(100)
model = gs.Gaussian(dim=2, var=1, len_scale=10)  # len_scale near zero
srf = gs.SRF(model, mode_no=100)
sample = srf((x, y), mesh_type='structured',seed=2)
plt.imshow(sample, interpolation='nearest')
plt.savefig('variogram_explained.png', dpi=300)
plt.colorbar()
plt.clf()


bin_center, gamma = gs.vario_estimate((x, y), sample, )
# fit the variogram with a stable model. (no nugget fitted)
plt.scatter(bin_center, gamma)
plt.ylim(0, 1.2)

# Add axis labels
plt.xlabel("Lag Distance", fontsize=14)
plt.ylabel("Semivariance", fontsize=14)
plt.show()
