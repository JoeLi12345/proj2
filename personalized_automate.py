import wesad_personalized
import statistics as stats
import sys

def ssl(subject):
	print("WESAD PERSONALIZED SUBJECT", subject)
	wesad_personalized.init_wandb(name=f"P_{subject}")
	wesad_personalized.train(subject)
	acc, acc1, f1, f11 = wesad_personalized.test(subject)
	print(acc, acc1, f1, f11)

for i in range(15):
	ssl(i)
