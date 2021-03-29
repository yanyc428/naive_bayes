import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('-l', '--local', action="store_true", help='choose whether from a local model')
parser.add_argument('-t', '--train', type=int, required=True, help='choose a train set size')
parser.add_argument('-e', '--evaluate', type=int, required=True, help='choose a evaluation set size')
args = parser.parse_args()


if __name__ == '__main__':
    from model import Model
    from data import load_data
    if args.local:
        model = Model.load("model_" + str(args.train) + ".pkl")
    else:
        model = Model(args.train, save=True, path="model_" + str(args.train) + ".pkl")
    print(model.evaluate(load_data(1)))
