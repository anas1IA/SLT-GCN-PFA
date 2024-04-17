from asl_data import ASLDataset



if __name__ == "__main__":

    dataset = ASLDataset()

    data = dataset[400]

    print(data.x)
    print(data.y)
    print(data.edge_index)
