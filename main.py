import models.simsiam_builder

batch_size = 16
lr = 0.1


'''
Create and train a simsiam model
'''
print("Creating SimSiam model...")
simsiam =  models.simsiam_builder.SimSiam()

# infer learning rate before changing batch size
init_lr = args.lr * args.batch_size / 256
# define loss function (criterion) and optimizer
criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)