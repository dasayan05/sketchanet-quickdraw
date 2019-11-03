import os, numpy as np
import matplotlib.pyplot as plt
from quickdraw_bin_parser import unpack_drawings
import json

def create_drawing( Q, save_name ):
    image_strokes = Q['image']
    # produces a numpy image with the drawing in it
    fig = plt.figure()
    # breakpoint()
    for stroke_xs, stroke_ys in image_strokes:
        stroke = np.vstack((np.array(stroke_xs), np.array(stroke_ys))).T
        # stroke = stroke / np.sqrt(stroke[:,0]**2 + stroke[:,1]**2).max()
        plt.plot(stroke[:,0], stroke[:,1])
    plt.xlim(0, 255); plt.ylim(0, 225)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.close()

def write_as_json( Q, save_name ):
    del Q['countrycode'] # bytes not serializable
    with open(save_name + '.json', 'w') as f:
        json.dump(Q, f)

def main( args ):
    # saver() should have a signature: saver(Q, filename_wo_ext)
    saver = create_drawing

    # create 'train' and 'test' folder
    os.mkdir(os.path.join(args.outdir, 'train'))
    os.mkdir(os.path.join(args.outdir, 'test'))

    # main entry point
    category_list = os.listdir(args.indir)
    np.random.shuffle(category_list)
    for i_cat, binfile in enumerate(category_list):
        if i_cat > (args.classes - 1):
            break
        else:
            print(binfile)
        
        filename = os.path.join(args.indir, binfile)
        os.mkdir(os.path.join(args.outdir, 'train', binfile.split('.')[0]))
        os.mkdir(os.path.join(args.outdir, 'test', binfile.split('.')[0]))
        
        for count, drawing in enumerate(unpack_drawings(filename)):
            # do something with the drawing
            save_path = os.path.join(args.outdir, 'train' if np.random.rand() < args.train_proportion else 'test',
                binfile.split('.')[0], str(count + 1))
            saver(drawing, save_path)
            if count + 1 >= args.n_eachclass:
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="""
        Prepares the QuickDraw dataset for ease of use. It does the following:
        1. Reads the '.bin' files of quickdraw from 'indir'
        2. Picks up 'n_eachclass' samples from each class
        3. Picks up 'classes' categories
        4. Splits the total 'n_eachclass' samples into 'train_proportion' proportion into train and test
    """)

    parser.add_argument('--indir', '-i', type=str, required=True, help='folder containing quickdraw .bin files')
    parser.add_argument('--outdir', '-o', type=str, required=True, help='folder (empty) to output rasterized dataset')
    parser.add_argument('--classes', '-c', type=int, required=False, default=10, help='How many classes to prepare?')
    parser.add_argument('--n_eachclass', '-n', type=int, required=False, default=1500, help='how many samples for each class?' )
    parser.add_argument('--train_proportion', '-p', type=float, required=False, default=0.85, help='How much proportion of "n_eachclass" you want as training data?')
    args = parser.parse_args()

    main( args )