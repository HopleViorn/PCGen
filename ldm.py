import os
from utils import *

# Parse input augments
args = get_args_ldm()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

from dataset import *
from trainer import *

def run(args):
    # Initialize dataset and trainer
    if args.option == 'surfpos':
        train_dataset = SurfPosData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfPosData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = SurfPosTrainer(args, train_dataset, val_dataset)

    elif args.option == 'surfz':
        train_dataset = SurfZData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfZData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = SurfZTrainer(args, train_dataset, val_dataset)

    elif args.option == 'edgepos':
        train_dataset = EdgePosData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = EdgePosData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = EdgePosTrainer(args, train_dataset, val_dataset)

    elif args.option == 'edgez':
        train_dataset = EdgeZData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = EdgeZData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = EdgeZTrainer(args, train_dataset, val_dataset)

    elif args.option == 'cond_surfpos':
        train_dataset = CondSurfPosData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = CondSurfPosData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = CondSurfPosTrainer(args, train_dataset, val_dataset)

    elif args.option == 'cond_surfz':
        train_dataset = CondSurfZData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = CondSurfZData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = CondSurfTrainer(args, train_dataset, val_dataset)

    elif args.option == 'cond_edgepos':
        train_dataset = CondEdgePosData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = CondEdgePosData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = CondEdgePosTrainer(args, train_dataset, val_dataset)

    elif args.option == 'cond_edgez':
        train_dataset = CondEdgeZData(args.data, getattr(args, 'list', None), validate=False, aug=args.data_aug, args=args)
        val_dataset = CondEdgeZData(args.data, getattr(args, 'list', None), validate=True, aug=False, args=args)
        ldm = CondEdgeZTrainer(args, train_dataset, val_dataset)

    else:
        print(args.option)
        assert False, 'please choose between [surfpos, surfz, edgepos, edgez, cond_surfpos, cond_surfz]'

    print('Start training...')
    
    # Main training loop
    for _ in range(args.train_nepoch):

        # Train for one epoch
        ldm.train_one_epoch()        

        # Evaluate model performance on validation set
        if ldm.epoch % args.test_nepoch == 0:
            ldm.test_val()

        # save model
        if ldm.epoch % args.save_nepoch == 0:
            ldm.save_model()

    return


if __name__ == "__main__":
    run(args)