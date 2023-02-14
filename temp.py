import argparse

def main():
    parser = argparse.ArgumentParser(description='Train radar detector')
    parser.add_argument('--model', help = 'model name', default='Crossatten',
                        choices=['RODNet', 'Crossatten', 'RAMP'])

    parser.add_argument('--gauss', help = 'Type of gauss', default='Bivar',
                        choices=['Gauss', 'Bivar'])
    
    parser.add_argument('--frame', help = 'Number of past frames (max 5)', type=int,
                         default=1, choices=range(1,6))
    parser.add_argument('--no_class', help='Number of classes', default=3)

    parser.add_argument('--co', help = 'Center offset loss activate', type=bool, default=1)
    parser.add_argument('--oren', help = 'Heading estimation', type=bool, default=0)
    parser.add_argument('--config', help='Config file path', default='config.yaml')
    parser.add_argument('--device', help='Either cuda:0 or CPU', default='cpu')
    parser.add_argument('--tag', help='an unique tag to save results',
                        default='exp1')
    parser.add_argument('--data_dir', help='Datset directory', default='dataset')

    args = parser.parse_args()

if __name__ == '__main__':
    main()