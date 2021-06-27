import argparse

def masuk(a,b):
    return a+b

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, help='help me', required=True)
    parser.add_argument('-y',type=int, help='help me', required=True)
    args = parser.parse_args()

    # val = 1
    g = args.g
    print(type(g))
    y = args.y

    print(masuk(g,y))