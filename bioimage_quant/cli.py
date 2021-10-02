import argparse

def cli():

    parser = argparse.ArgumentParser()
    parser.add_argument('test')

    args = parser.parse_args()
    print(args.test)

if __name__ == "__main__":
    cli()
