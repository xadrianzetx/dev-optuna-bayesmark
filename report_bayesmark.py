import os


def read_partials() -> None:
    print(os.listdir("partial"))
    open(os.path.join("report", "payload.txt"), "w").close()


if __name__ == "__main__":
    read_partials()
