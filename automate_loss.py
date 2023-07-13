import wesad_loss
import sys

def automate(subject):
	wesad_loss.init_wandb(name=f'S{subject}')
	results = [0, 0.1, 0.2, 0.3]
	results = [0.4, 0.5, 0.6, 0.7]
	results = [0.8, 0.9, 1]
	pr = []
	for i in range(len(results)):
		wesad_loss.train(subject, weight=results[i])
		acc, acc1, f1, f11 = wesad_loss.test(subject)
		pr.append([acc, acc1, f1, f11])
	for i in range(len(results)):
		print("subject ", subject, "weight", results[i], "accuracy", pr[i][0], pr[i][1], pr[i][2], pr[i][3])

automate(14)
