import Final_Project.preprocess as pm

DATASET_PATH = "Final_Project/Dataset/dataset_for_traffic_project_assignment/training_dataset.csv"


def main():
    df = pm.load_dataset(DATASET_PATH)
    print(df.shape)
    print(df.info())
    df = pm.engineer_features(df)
    print(df.info())
    dataframe = pm.prepare_model_dataframe(df)
    corr_matrix = pm.compute_compact_correlation(dataframe)
    pm.plot_correlation_heatmap(corr_matrix)


if __name__ == "__main__":
    main()
