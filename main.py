from regressor import Regressor
import matplotlib.pyplot as plt
import numpy as np

teste = Regressor()
teste.fit(1, 26)
teste.derivada(np.random.random_sample(), np.random.random_sample())
teste.predict()
plt.scatter(teste.X_train, teste.Y_train)
plt.title('House pricing')
plt.xlabel('house area')
plt.ylabel('house price')
plt.savefig('train.png')
plt.plot(teste.X_train, teste.Y_test)
plt.savefig('results.png')