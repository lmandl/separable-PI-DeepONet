
from models.deep_onet import DeepONet
from data.data_generator import generate_data

if __name__ == '__main__':
    print('Project in Development')
    # Generate data
    train_data, test_data = generate_data()

    # Initialize model
    model = DeepONet()

    # Train model
    model.train(train_data)

    # Test model
    model.test(test_data)