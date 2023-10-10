import argparse
import json
from train import ModelTrainer
from evaluate import ModelTester

def main(args):
    # Load configurations
    with open("config.json", "r") as file:
        config = json.load(file)

    if args.mode == 'train':
        trainer = ModelTrainer(config)
        trainer.train(args.epochs)
    elif args.mode == 'eval':
        tester = ModelTester(config_path="config.json",
                             model_weights_path=config['models']['weights_path'],
                             test_data_path=config['data']['train'],  # assuming you want to use train_data here
                             stopwords_path=config['data']['stopwords'],
                             vncore_path=config['data']['vncorenlp'])
        tester.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Control training and evaluation of the model from here')
    parser.add_argument('mode', type=str, choices=['train', 'eval'], help='Whether to train or evaluate the model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train for (only used in train mode)')
    args = parser.parse_args()

    main(args)
