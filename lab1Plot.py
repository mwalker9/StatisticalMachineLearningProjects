%matplotlib inline
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
data = pandas.read_csv("store_train.csv", low_memory=False)
data = data[data.Store == 1]
fig=plt.figure(figsize=(16,4))
fig.suptitle('Store 1 Sales')
plt.plot_date(pandas.to_datetime(data["Date"]), data["Sales"], ls="solid", Figure=fig, Marker=None)
ax = plt.gca()
ax.set_xlabel("Date")
ax.set_ylabel("Sales($)")
blue_patch = mpatches.Patch(color='blue', label="Sales for store 1")
plt.legend(handles=[blue_patch])
plt.tight_layout()
plt.show()