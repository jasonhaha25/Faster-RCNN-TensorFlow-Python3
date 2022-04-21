if __name__ == "__main__":
	with open("./times.txt", "r") as f:
		times = [float(t[:-1]) for t in f.readlines()]
		print("Total time: ", sum(times))
		print("Average time: ", sum(times) / len(times))