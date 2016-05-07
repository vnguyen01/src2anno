anno = []
with open("data/en-django/all.anno", "r") as f:
	for line in f:
		line = line.strip()
		line = line.replace(".", " . ")
		anno.append(line)
f.close()
print len(anno)
with open("data/en-django/all3.anno", "w") as f:
	for line in anno:
		f.write(line + "\n")
f.close()