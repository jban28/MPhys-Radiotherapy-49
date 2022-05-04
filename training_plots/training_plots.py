import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def smooth(array, smoothing):
  smoothed = np.convolve(array, np.ones(smoothing), "valid")/smoothing
  return smoothed

data = np.genfromtxt("training.csv", dtype="str", delimiter=",")[1:50]
data = data.astype(float)

smth = 5 # Must be odd integer

epoch = data[:,0]
tr_loss = data[:,1]
val_loss = data[:,2]
acc = data[:,3]
sens = data[:,4]
spec = data[:,5]

# epoch_sm = epoch[int(smth/2):-int(smth/2)]
epoch_sm = epoch[smth-1:]
tr_loss_sm = smooth(tr_loss,smth)
val_loss_sm = smooth(val_loss,smth)

plt.rcParams["font.family"] = "serif"
plt.plot(epoch, tr_loss, color="blue", linestyle="dotted")
plt.plot(epoch_sm,tr_loss_sm, color="blue", label="Training")
plt.plot(epoch, val_loss, color="red", linestyle="dotted")
plt.plot(epoch_sm,val_loss_sm, color="red", label="Validation")
plt.xlabel("Training Epoch")
plt.legend()
plt.ylabel("Loss")
plt.savefig("loss_plot.png")
plt.savefig("loss_plot.pdf")
plt.clf()

plt.plot(epoch, sens)
plt.plot(epoch, spec)
plt.xlabel("Training Epoch")
plt.ylabel("Parameter Value")

plt.savefig("sens_spec.png")