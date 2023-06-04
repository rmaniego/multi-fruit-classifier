import os

start = 335

for filename in os.listdir("."):
    if not os.path.isfile(filename) or filename.split(".")[-1].lower() != "jpg":
        continue

    new_filename = "OR" + str(start).rjust(4, "0") + ".jpg"
    if not os.path.exists(new_filename):
        os.rename(filename, new_filename)
        start += 1