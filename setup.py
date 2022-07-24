














if __name__ == "__main__":
    parser = ArgumentParser("Train Move Classifier on Feature")

    # Paths to Data
    

    #Setup Arguments
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--balanced", type=int, default=0)
    parser.add_argument("--shrink", type=float, default=1.0)
    parser.add_argument("--subset", type=float, default=1.0)

    args = parser.parse_args()

    BALANCED = False
    if args.balanced == 1:
        BALANCED = True
    
    INPUTSHAPE = (int(640*args.shrink), int(360*args.shrink), 3)

    # Call Main
    main(args.feature, args.batchsize, args.epochs, BALANCED, INPUTSHAPE, args.subset)