import torch, os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils import tensorboard as tb
from sketchanet import SketchANet

def main( args ):
    quickdraw_trainds = ImageFolder(args.traindata, transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()]))
    quickdraw_traindl = DataLoader(quickdraw_trainds, batch_size=args.batch_size, pin_memory=True, num_workers=os.cpu_count(),
        shuffle=True, drop_last=True)

    quickdraw_testds = ImageFolder(args.testdata, transform=Compose([Resize([225, 225]), Grayscale(), ToTensor()]))
    quickdraw_testdl = DataLoader(quickdraw_testds, batch_size=args.batch_size * 2, pin_memory=True, num_workers=os.cpu_count(),
        shuffle=True, drop_last=True)

    model = SketchANet(num_classes=args.num_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    
    crit = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())

    # Tensorboard stuff
    writer = tb.SummaryWriter('./logs')

    count = 0
    for e in range(args.epochs):
        for i, (X, Y) in enumerate(quickdraw_traindl):
            # Binarizing 'X'
            X[X < 1.] = 0.; X = 1. - X

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optim.zero_grad()

            output = model(X)
            loss = crit(output, Y)
            
            if i % args.print_interval == 0:
                print(f'[Training] {i}/{e}/{args.epochs} -> Loss: {loss.item()}')
                writer.add_scalar('train-loss', loss.item(), count)
            
            loss.backward()
            optim.step()

            count += 1

        correct, total = 0, 0
        for i, (X, Y) in enumerate(quickdraw_testdl):
            # Binarizing 'X'
            X[X < 1.] = 0.; X = 1. - X

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            output = model(X)
            _, predicted = torch.max(output, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            accuracy = (correct / total) * 100
        
        print(f'[Testing] -/{e}/{args.epochs} -> Accuracy: {accuracy} %')
        writer.add_scalar('test-accuracy', accuracy/100., e)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata', type=str, required=True, help='root of the train datasets')
    parser.add_argument('--testdata', type=str, required=True, help='root of the test datasets')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='Batch size')
    parser.add_argument('-c', '--num_classes', type=int, required=False, default=10, help='Number of classes for the classification task')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=100, help='No. of epochs')
    parser.add_argument('-i', '--print_interval', type=int, required=False, default=10, help='Print loss after this many iterations')
    args = parser.parse_args()

    main( args )