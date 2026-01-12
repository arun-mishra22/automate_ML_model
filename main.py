from data_loader_new import load_and_prepare_dataset

df, X, y, summary = load_and_prepare_dataset(
    file_path = "C:/Users/arun4/OneDrive/Desktop/auto model/Telco-Customer-Churn-1.csv",
    target_col="Churn"
)

print(summary)
print(X.shape, y.shape)

