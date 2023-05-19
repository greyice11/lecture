import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('train.csv')
df.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS'], axis=1, inplace=True)
df = df.astype(float)
y = df['TARGET(PRICE_IN_LACS)']
X = df.drop(columns=['TARGET(PRICE_IN_LACS)'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

model1 = LinearRegression().fit(X_train, y_train)

new_data = pd.read_csv('test.csv')
new_data.drop(['POSTED_BY', 'BHK_OR_RK', 'ADDRESS'], axis=1, inplace=True)
new_data = new_data.astype(float)
new_data_pred = model1.predict(new_data)
result_df = pd.concat([new_data, pd.DataFrame({'Predicted Price': new_data_pred})], axis=1)
result_df.to_excel('predicted_prices.xlsx', index=False)

plt.scatter(y_test, model1.predict(X_test))
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

plt.hist(new_data_pred, bins=50)
plt.xlabel('Predicted Prices')
plt.ylabel('Frequency')
plt.title('Predicted Prices Distribution')
plt.show()