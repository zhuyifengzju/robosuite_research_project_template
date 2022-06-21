












bcolors = {"blue": '\033[94m', "end": '\033[0m', "green": '\033[92m', "yellow": '\033[93m', "cyan": '\033[96m', "red": '\033[91m'}

def color_print(color=None, *args):
    if color is None:
        print(*args)
    else:
        print(bcolors[color], *args, bcolors["end"])

