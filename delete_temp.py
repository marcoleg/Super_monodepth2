import os


def main():
    for dire in os.listdir(os.path.curdir):
        if os.path.isdir(dire) and dire == 'tmp':
            for sub_dir in os.listdir(os.path.join(os.path.curdir, dire)):
                if sub_dir == 'mono_model':
                    for sub_sub_dir in os.listdir(os.path.join(os.path.curdir, dire, sub_dir)):
                        if sub_sub_dir == 'train' or sub_sub_dir == 'val':
                            for event_file in os.listdir(os.path.join(os.path.curdir, dire, sub_dir, sub_sub_dir)):
                                os.remove(os.path.join(os.path.curdir, dire, sub_dir, sub_sub_dir, event_file))
    print("Old Temp Tensorboard Event Files Removed!")


if __name__ == '__main__':
    main()
