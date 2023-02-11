import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import DataGenerator
from meta import Meta
import torch
import numpy as np

def monitor_training(H, K, N):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H["train_loss"])), H["train_loss"], label="train_loss")
    plt.plot(np.arange(0, len(H["test_loss"])), H["test_loss"], label="test_loss")
    plt.title("Training and Testing Losses")
    plt.xlabel("Iter #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses_K{}_N{}.png".format(K, N))
    plt.close()

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(H["train_acc"])), H["train_acc"], label="train_acc")
    plt.plot(np.arange(0, len(H["test_acc"])), H["test_acc"], label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Iter #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_K{}_N{}.png".format(K, N))
    plt.close()

def main():
	outer_lr = 1e-3
	inner_lr = 0.4
	n_way = 5
	k_support = 1
	k_query = 15
	batch_size = 32
	num_workers = 2 #4
	epoch = 40000
	
	H = {}
	H["train_loss"] = []
	H["test_loss"] = []
	H["train_acc"] = []
	H["test_acc"] = []

	device = torch.device("cuda") # "cuda" or "cpu"

	maml = Meta(outer_lr, inner_lr, n_way, k_support, k_query, batch_size).to(device)

	# Create Data Generator
	train_iterable = DataGenerator(n_way, k_support, k_query, batch_type="train")
	test_iterable = DataGenerator(n_way, k_support, k_query, batch_type="test")

	train_loader = iter(torch.utils.data.DataLoader(train_iterable, batch_size=batch_size, 
		num_workers=num_workers, pin_memory=True,))

	test_loader = iter(torch.utils.data.DataLoader(test_iterable, batch_size=batch_size, 
		num_workers=num_workers, pin_memory=True,))

	for step in range(epoch):
		# x_support, y_support, x_query, y_query = train_loader.next()
		x_support, y_support, x_query, y_query = next(train_loader)

		x_support = x_support.to(device)
		y_support = y_support.to(device)
		x_query = x_query.to(device)
		y_query = y_query.to(device)

		accs, loss = maml(x_support, y_support, x_query, y_query)

		if step % 500 == 0:

			print("Step: {}, training acc: {}, training loss {}".format(step, accs, loss))
			H["train_acc"].append(accs)
			H["train_loss"].append(loss)

			accs = []
			losses = []

			for _ in range(1000//batch_size):

				# x_support, y_support, x_query, y_query = test_loader.next()
				x_support, y_support, x_query, y_query = next(test_loader)

				x_support = x_support.to(device)
				y_support = y_support.to(device)
				x_query = x_query.to(device)
				y_query = y_query.to(device)

				# split to single task
				for x_support_one, y_support_one, x_query_one, y_query_one in \
					zip(x_support, y_support, x_query, y_query):

					test_acc, test_loss = maml.finetuning(x_support_one, y_support_one, x_query_one, y_query_one)

					accs.append(test_acc)
					losses.append(test_loss)

			test_acc = np.array(accs).mean(axis=0).astype(np.float16)
			test_loss = np.array(losses).mean(axis=0).astype(np.float16)

			print("test acc: {}, test loss {}".format(test_acc, test_loss))
			H["test_acc"].append(test_acc)
			H["test_loss"].append(test_loss)

			monitor_training(H, k_support, n_way)

		if step % 8000 == 0:
			print("saving checkpoint...")
			torch.save(maml.state_dict(), "maml_run1_checkpoint_{}.pt".format(step))

	print("saving trained model...")
	torch.save(maml.state_dict(), "maml_run1.pt")

if __name__ == '__main__':
	main()







