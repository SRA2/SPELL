import os
import argparse
from data_loader import AVADataset

parser = argparse.ArgumentParser(description='generate_graph')
parser.add_argument('--feature', type=str, default='resnet18-tsm-aug', help='name of the features')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--time_edge', type=float, default=0.9, help='time threshold')
parser.add_argument('--cross_identity', type=str, default='cin', help='whether to allow cross-identity edges')
parser.add_argument('--edge_weight', type=str, default='fsimy', help='how to decide edge weights')


def main():
    args = parser.parse_args()

    # dict that stores graph parameters
    graph_data={}
    graph_data['numv'] = args.numv
    graph_data['skip'] = graph_data['numv']                ## if 'skip' < 'numv' then there will be overlap between graphs of length numv-skip
    graph_data['time_edge'] = args.time_edge               ## time support of the graph
    graph_data['cross_identity'] = args.cross_identity     ## 'ciy' allows cross-identity edges, 'cin': No cross-idenity edges
    graph_data['edge_weight'] = args.edge_weight           ## fsimn vs fsimy as above

    # target path for storing graphs
    tpath_key = os.path.join('graphs', '{}_{}_{}_{}_{}'.format(args.feature, graph_data['numv'], graph_data['time_edge'], graph_data['cross_identity'], graph_data['edge_weight']))

    for mode in ['train', 'val']:
        # specifies location of the features within feature path
        dpath_mode = os.path.join('features', args.feature, '{}_forward'.format(mode), '*.csv')

        # specifies location of the graphs
        tpath_mode = os.path.join(tpath_key, mode)

        graph_gen(dpath_mode, tpath_mode, graph_data, mode)


# function that takes input of feature path and target path for storing graphs and creates graphs using the dataloader AVADataset
def graph_gen(dpath, tpath, graph_data, mode, cont=0):
    os.makedirs(tpath, exist_ok=True)
    Fdataset = AVADataset(dpath, graph_data, cont, tpath, mode)


if __name__ == '__main__':
    main()
